"""
Extract behavioral fingerprints for each (game, player) pair from v4p_games.jsonl.

Behavioral features over the first OBS_WINDOW turns:
  - launch_rate, idle_rate
  - avg_ships, median_ships, p90_ships, fleet_size_gini (aggressiveness)
  - n_distinct_origins, origin_concentration (concentration of launches)
  - early_planet_gain, early_ship_growth (expansion speed)
  - fleet_to_total_ratio (how much sits in flight vs garrison)
  - centrality_drift (expand outward vs hunker inward)
  - aggression_radius (avg ships per launch / avg ships owned)

Output: one row per (game_idx, player_idx) →
  data/opp_behaviors.jsonl
"""
import argparse, json, math, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "v4p_games.jsonl"
OUT = ROOT / "data" / "opp_behaviors.jsonl"

OBS_WINDOW = 30  # turns of observation


def gini(xs):
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    s = sum(xs)
    if s < 1e-9:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs):
        cum += (i + 1) * x
    return (2 * cum) / (n * s) - (n + 1) / n


def behavior_features(traj, num_players):
    """
    traj: list of {state, action} for one player over the whole game.
    Returns dict of behavioral features computed from first OBS_WINDOW turns.
    """
    window = traj[:OBS_WINDOW]
    if not window:
        return None

    # Action-derived features
    all_actions = []          # flatten: each launch is a single record
    turns_with_action = 0
    turns_total = len(window)
    distinct_origins = set()
    ship_sizes = []
    for turn in window:
        acts = turn.get("action") or []
        if acts:
            turns_with_action += 1
        for a in acts:
            # action format: [from_planet_id, angle, ships]
            if len(a) >= 3:
                from_id, angle, ships = a[0], a[1], a[2]
                all_actions.append(a)
                ship_sizes.append(ships)
                distinct_origins.add(from_id)

    launch_rate = len(all_actions) / max(1, turns_total)
    idle_rate = 1 - turns_with_action / max(1, turns_total)

    if ship_sizes:
        avg_ships = statistics.mean(ship_sizes)
        med_ships = statistics.median(ship_sizes)
        p90_ships = sorted(ship_sizes)[int(len(ship_sizes)*0.9)]
        ship_gini = gini(ship_sizes)
    else:
        avg_ships = med_ships = p90_ships = ship_gini = 0.0

    # State-derived features (compact_state has 'my' summary)
    s0 = window[0]["state"]["my"]
    sN = window[-1]["state"]["my"]
    planet_gain = sN["planets"] - s0["planets"]
    ship_growth = sN["ships"] / max(1, s0["ships"])

    # Fleet-to-total: ships in flight / total ships, avg over later turns
    fleet_to_total = []
    for turn in window[5:]:
        my = turn["state"]["my"]
        total = my["ships"]
        if total > 0:
            fleet_to_total.append(my["fleet_ships"] / total)
    avg_fleet_ratio = statistics.mean(fleet_to_total) if fleet_to_total else 0.0

    # Centrality: my_centrality is dist from board center of my planets
    # high → far from center (edge play); low → near center
    centrality_0 = s0["centrality"]
    centrality_N = sN["centrality"]
    centrality_drift = centrality_N - centrality_0

    # Aggression radius: avg ships per launch / avg ships per planet
    # high → I send big fleets relative to my stock (committed attacker)
    avg_per_planet = (sN["ships"] - sN["fleet_ships"]) / max(1, sN["planets"])
    aggression = avg_ships / max(1.0, avg_per_planet) if avg_ships else 0.0

    n_distinct_origins = len(distinct_origins)
    distinct_origin_frac = n_distinct_origins / max(1, sN["planets"])

    return {
        "launch_rate": round(launch_rate, 3),
        "idle_rate": round(idle_rate, 3),
        "avg_ships": round(avg_ships, 2),
        "med_ships": round(med_ships, 2),
        "p90_ships": round(p90_ships, 2),
        "ship_gini": round(ship_gini, 3),
        "n_distinct_origins": n_distinct_origins,
        "distinct_origin_frac": round(distinct_origin_frac, 3),
        "early_planet_gain": planet_gain,
        "early_ship_growth": round(ship_growth, 3),
        "avg_fleet_ratio": round(avg_fleet_ratio, 3),
        "centrality_drift": round(centrality_drift, 2),
        "aggression": round(aggression, 3),
        "n_launches": len(all_actions),
    }


def main():
    if not DATA.exists():
        print(f"ERROR: {DATA} not found")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_out = 0
    t0 = time.time()
    with open(DATA) as fin, open(OUT, "w") as fout:
        for line in fin:
            g = json.loads(line)
            n_in += 1
            seed = g["seed"]
            agents = g["agents"]
            num_players = g.get("num_players", 4)
            traces = g["traces"]
            for p_idx_str, traj in traces.items():
                p_idx = int(p_idx_str)
                feats = behavior_features(traj, num_players)
                if feats is None:
                    continue
                feats.update({
                    "seed": seed,
                    "player_idx": p_idx,
                    "agent_name": agents[p_idx],  # GROUND TRUTH label
                    "num_players": num_players,
                    "is_winner": p_idx == g["winner"],
                    "game_steps": g.get("steps", 0),
                })
                fout.write(json.dumps(feats) + "\n")
                n_out += 1

    print(f"Read {n_in} games → wrote {n_out} (game, player) rows to {OUT}")
    print(f"Time: {time.time()-t0:.1f}s")

    # Quick label distribution
    from collections import Counter
    rows = [json.loads(l) for l in open(OUT)]
    print("\nAgent distribution:")
    for name, count in Counter(r["agent_name"] for r in rows).most_common():
        print(f"  {name:<18}  n={count}")


if __name__ == "__main__":
    main()
