"""
Extract shot outcomes from data/shot_games.jsonl (compact_state format).

For each (game, player, turn, action) tuple:
  - Identify target via ray-cast against planets
  - Look forward LOOKAHEAD turns
  - Label: did target become player-owned within LOOKAHEAD?

Output features per shot:
  - source_ships, source_prod
  - target_ships, target_prod, target_owner_type, target_dist
  - fleet_size, fleet_speed, eta, predicted_garrison_at_arrival, margin
  - global: my_planets, my_ships, opp_ships, step, num_players

Output: data/shot_outcomes_v2.jsonl
"""

import json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "shot_games.jsonl"
OUT = ROOT / "data" / "shot_outcomes_v2.jsonl"

LOOKAHEAD = 20  # turns ahead to check capture
MAX_SPEED = 6.0


def fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5


def find_target(state, source_id, angle):
    """Ray cast from source planet to find target."""
    src = None
    for p in state["p"]:
        if p[0] == source_id:
            src = p
            break
    if src is None:
        return None
    sx, sy = src[2], src[3]
    dx, dy = math.cos(angle), math.sin(angle)
    best = None
    best_proj = float("inf")
    for p in state["p"]:
        if p[0] == source_id:
            continue
        rx = p[2] - sx
        ry = p[3] - sy
        proj = rx * dx + ry * dy
        if proj <= 0:
            continue
        perp = abs(rx * (-dy) + ry * dx)
        if perp < 4.0 and proj < best_proj:
            best_proj = proj
            best = p
    if best is None:
        return None
    return src, best, best_proj


def shot_features(state, src, tgt, ships, player, num_players):
    sx, sy = src[2], src[3]
    tx, ty = tgt[2], tgt[3]
    dist = math.hypot(tx - sx, ty - sy)
    speed = fleet_speed(ships)
    eta = max(1, math.ceil(dist / speed))

    tgt_owner = tgt[1]
    if tgt_owner == -1:
        owner_type = 0  # neutral
    elif tgt_owner == player:
        owner_type = 2  # self
    else:
        owner_type = 1  # enemy

    if owner_type == 0:
        predicted_garrison = tgt[4]
    else:
        predicted_garrison = tgt[4] + tgt[5] * eta

    margin = ships - predicted_garrison

    my_planets = sum(1 for p in state["p"] if p[1] == player)
    my_ships = sum(p[4] for p in state["p"] if p[1] == player)
    opp_ships = sum(p[4] for p in state["p"] if p[1] != player and p[1] != -1)

    return [
        src[4],  # src_ships
        src[5],  # src_prod
        tgt[4],  # tgt_ships
        tgt[5],  # tgt_prod
        owner_type,
        dist,
        ships,
        speed,
        eta,
        predicted_garrison,
        margin,
        my_planets,
        my_ships,
        opp_ships,
        state["step"] / 500.0,
        num_players / 4.0,
        1.0 if num_players == 2 else 0.0,
    ]


FEATURE_NAMES = [
    "src_ships", "src_prod", "tgt_ships", "tgt_prod", "tgt_owner_type",
    "distance", "fleet_size", "fleet_speed", "eta", "predicted_garrison",
    "margin", "my_planets_count", "my_total_ships", "opp_total_ships",
    "step_progress", "num_players_norm", "is_2p",
]


def check_capture(traces, player, target_id, current_turn, lookahead):
    traj = traces[str(player)]
    for offset in range(1, lookahead + 1):
        ft = current_turn + offset
        if ft >= len(traj):
            return 0
        st = traj[ft]["state"]
        for p in st["p"]:
            if p[0] == target_id and p[1] == player:
                return 1
    return 0


def main():
    samples = 0
    cap_success = 0
    skipped = 0
    n_games = 0
    with open(DATA) as f_in, open(OUT, "w") as f_out:
        for line in f_in:
            game = json.loads(line)
            n_games += 1
            num_players = game["num_players"]
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
                        res = find_target(state, source_id, angle)
                        if res is None:
                            skipped += 1
                            continue
                        src, tgt, _ = res
                        # Skip reinforce (target is self)
                        if tgt[1] == p_idx:
                            continue
                        feats = shot_features(state, src, tgt, ships,
                                                p_idx, num_players)
                        label = check_capture(game["traces"], p_idx, tgt[0],
                                              turn_idx, LOOKAHEAD)
                        f_out.write(json.dumps({
                            "feats": feats,
                            "label": label,
                            "player": p_idx,
                            "did_win": int(p_idx == game["winner"]),
                        }) + "\n")
                        samples += 1
                        cap_success += label
    print(f"Games: {n_games}")
    print(f"Shots: {samples}")
    print(f"Capture success: {cap_success}/{samples} = {cap_success/max(1,samples):.1%}")
    print(f"Skipped: {skipped}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
