"""
Extract (shot_features, success) training data from round-robin game traces.

For each (game, player, turn, action) tuple, we know:
  - The player launched a fleet from source_planet at angle with ships ships
  - We can predict which target the fleet hits (ray cast against planets)
  - Look forward K turns: did target become owned by player AND have only the
    survivor ships? That's a successful capture.

Features per shot:
  - source_ships (before launch)
  - source_prod
  - target_ships (current)
  - target_prod
  - target_owner_type (0=neutral, 1=enemy, 2=self-reinforce)
  - distance
  - fleet_size
  - fleet_speed (computed)
  - expected_eta (turns to arrival)
  - my_total_planets (global state)
  - my_total_ships
  - opp_total_ships
  - step / 500

Label: 1 if capture succeeded within K=15 turns, else 0.

Output:
  data/shot_outcomes.jsonl
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "round_robin_games.jsonl"
OUT = ROOT / "data" / "shot_outcomes.jsonl"

MAX_SPEED = 6.0
LOOKAHEAD = 15  # turns ahead to check capture


def fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5


def find_target_planet(state, source_id, angle, src_x=None, src_y=None):
    """Ray cast: find which planet the fleet at this angle from source will hit.

    state is the player's-pov summary (has my_planets/opp_planets/neutral_planets
    each with id, x, y, ships, prod). Returns target planet dict or None."""
    all_planets = state["my_planets"] + state["opp_planets"] + state["neutral_planets"]
    src = None
    for p in state["my_planets"]:
        if p["id"] == source_id:
            src = p
            break
    if src is None:
        return None
    sx, sy = src["x"], src["y"]
    dx, dy = math.cos(angle), math.sin(angle)
    best = None
    best_proj = float("inf")
    for p in all_planets:
        if p["id"] == source_id:
            continue
        rx = p["x"] - sx
        ry = p["y"] - sy
        proj = rx * dx + ry * dy
        if proj <= 0:
            continue
        perp = abs(rx * (-dy) + ry * dx)
        # planet radius ~ 1+log(prod), say 1-3 units
        if perp < 4.0 and proj < best_proj:
            best_proj = proj
            best = p
    if best is None:
        return None
    return best, src, best_proj


def shot_features(state, src, target, ships, num_players, is_2p):
    """Build a fixed-size feature vector for a shot."""
    dist = math.hypot(target["x"] - src["x"], target["y"] - src["y"])
    speed = fleet_speed(ships)
    eta = max(1, math.ceil(dist / speed))

    # Target owner classification
    if "owner" in target:  # enemy
        target_owner_type = 1
    elif target in state["my_planets"]:
        target_owner_type = 2  # reinforce
    else:
        target_owner_type = 0  # neutral

    # Predicted target ships at arrival
    if target_owner_type == 0:
        predicted_garrison = target["ships"]
    else:
        predicted_garrison = target["ships"] + target["prod"] * eta

    margin = ships - predicted_garrison

    my_total_ships = sum(p["ships"] for p in state["my_planets"])
    opp_total_ships = sum(p["ships"] for p in state["opp_planets"])
    my_planets_count = len(state["my_planets"])

    return [
        src["ships"],
        src["prod"],
        target["ships"],
        target["prod"],
        target_owner_type,
        dist,
        ships,
        speed,
        eta,
        predicted_garrison,
        margin,
        my_planets_count,
        my_total_ships,
        opp_total_ships,
        state["step"] / 500.0,
        1.0 if is_2p else 0.0,
        num_players / 4.0,
    ]


FEATURE_NAMES = [
    "src_ships", "src_prod", "tgt_ships", "tgt_prod", "tgt_owner_type",
    "distance", "fleet_size", "fleet_speed", "eta", "predicted_garrison",
    "margin", "my_planets_count", "my_total_ships", "opp_total_ships",
    "step", "is_2p", "num_players_norm",
]


def check_capture(player, target_id, traces, current_turn, lookahead):
    """Did the target become owned by player within lookahead turns?

    Returns 1 if successful capture (at least once), 0 otherwise.
    """
    # We need to look at the player's own trace at future turns
    if str(player) not in traces:
        return None
    my_trace = traces[str(player)]
    for offset in range(1, lookahead + 1):
        future_turn = current_turn + offset
        if future_turn >= len(my_trace):
            break
        state = my_trace[future_turn]["state"]
        # Was the target now ours?
        for p in state["my_planets"]:
            if p["id"] == target_id:
                return 1
    return 0


def main():
    if not DATA.exists():
        print(f"No data at {DATA}")
        return

    samples = 0
    cap_success = 0
    skipped = 0
    n_games = 0
    with open(DATA) as f_in, open(OUT, "w") as f_out:
        for line in f_in:
            game = json.loads(line)
            if "traces" not in game:
                continue
            n_games += 1
            num_players = game["num_players"]
            is_2p = num_players == 2
            for p_idx_str, traj in game["traces"].items():
                p_idx = int(p_idx_str)
                for turn_idx, turn in enumerate(traj):
                    action = turn["action"]
                    if not action:
                        continue
                    state = turn["state"]
                    for mv in action:
                        if not isinstance(mv, list) or len(mv) != 3:
                            continue
                        source_id, angle, ships = mv
                        result = find_target_planet(state, source_id, angle)
                        if result is None:
                            skipped += 1
                            continue
                        target, src, _ = result
                        # Don't train on reinforce moves (target is my own)
                        if target in state["my_planets"]:
                            continue
                        # Build features
                        feats = shot_features(state, src, target, ships,
                                                num_players, is_2p)
                        # Check capture outcome
                        label = check_capture(p_idx, target["id"],
                                              game["traces"], turn_idx, LOOKAHEAD)
                        if label is None:
                            continue
                        f_out.write(json.dumps({
                            "feats": feats,
                            "label": label,
                            "game_winner": game["winner"],
                            "player": p_idx,
                            "did_player_win": int(p_idx == game["winner"]),
                        }) + "\n")
                        samples += 1
                        if label:
                            cap_success += 1
    print(f"Games processed: {n_games}")
    print(f"Shots extracted: {samples}")
    print(f"Skipped (no target found): {skipped}")
    print(f"Success rate (true positives): {cap_success}/{samples} = "
          f"{cap_success/max(1,samples):.1%}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
