"""
Collect games across MANY seeds (= many maps) for clustering analysis.

Different from round_robin: emphasizes SEED diversity over pairing diversity.
With 4 strong agents and 30 seeds we get 30 unique maps × multiple pairings
per map = enough data for clustering + per-cluster win analysis.

Usage:
  .venv/bin/python -u scripts/collect_diverse.py [seeds=30]
"""

import argparse, json, logging, math, os, sys, time, itertools
from collections import Counter
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"
DATA = ROOT / "data" / "diverse_games.jsonl"


def ref(name):
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    return str(OPP_DIR / f"{name}.py")


def extract_map_features(obs):
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    ang_vel = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)
    homes = [p for p in planets if p[1] in (0, 1, 2, 3)]
    neutrals = [p for p in planets if p[1] == -1]
    prod_counter = Counter(p[6] for p in neutrals)
    rotating = sum(1 for p in planets if math.hypot(p[2]-50, p[3]-50) + p[4] < 50)
    dist_home_to_nearest = float("inf")
    if homes and neutrals:
        h = homes[0]
        dist_home_to_nearest = min(math.hypot(h[2]-n[2], h[3]-n[3]) for n in neutrals)
    avg_neutral_ships = sum(p[5] for p in neutrals) / max(1, len(neutrals))
    max_neutral_ships = max((p[5] for p in neutrals), default=0)
    total_n = sum(prod_counter.values())
    entropy = 0
    for c in prod_counter.values():
        p = c / total_n
        if p > 0:
            entropy -= p * math.log(p)
    return {
        "angular_velocity": round(ang_vel, 5),
        "num_planets": len(planets),
        "num_neutrals": len(neutrals),
        "rotating_planets": rotating,
        "static_planets": len(planets) - rotating,
        "prod_distribution": dict(prod_counter),
        "max_prod": max((p[6] for p in neutrals), default=0),
        "dist_home_to_nearest_neutral": round(dist_home_to_nearest, 2),
        "avg_neutral_ships": round(avg_neutral_ships, 1),
        "max_neutral_ships": max_neutral_ships,
        "prod_entropy": round(entropy, 3),
    }


def run_game(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run([ref(a) for a in agents])
    except Exception as e:
        return None, str(e)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r and r > 0]
    if not winners:
        return {"draw": True, "agents": agents, "seed": seed}, None
    map_features = extract_map_features(env.steps[0][0].observation)
    return {
        "agents": agents,
        "seed": seed,
        "winner": winners[0],
        "winner_agent": agents[winners[0]],
        "num_players": len(agents),
        "steps": len(env.steps),
        "rewards": rewards,
        "map_features": map_features,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=30)
    ap.add_argument("--agents", nargs="+",
                    default=["main", "lb1224", "mlhybrid", "structured",
                             "proto1000", "orbitbotnext"])
    args = ap.parse_args()

    if DATA.exists():
        DATA.unlink()

    pairings = list(itertools.permutations(args.agents, 2))
    total = args.seeds * len(pairings)
    print(f"Plan: {len(pairings)} pairings × {args.seeds} seeds = {total} games")
    print(f"Agents: {args.agents}")

    t0 = time.time()
    saved = err = draws = 0
    f_out = open(DATA, "a")
    for seed in range(args.seeds):
        for lineup in pairings:
            res, e = run_game(list(lineup), seed)
            if e:
                err += 1
                continue
            if res.get("draw"):
                draws += 1
                continue
            f_out.write(json.dumps(res) + "\n")
            f_out.flush()
            saved += 1
        elapsed = time.time() - t0
        eta = elapsed / max(0.001, (seed + 1)/args.seeds) - elapsed
        print(f"[seed {seed+1}/{args.seeds}] saved={saved} err={err} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    f_out.close()
    print(f"\nDone: {saved} games at {DATA}")


if __name__ == "__main__":
    main()
