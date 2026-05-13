"""
Round-robin tournament: every (agent_a, agent_b) pair × N seeds in 2P, plus
sampled 4P lineups. Captures per-game:
  - map features (extracted from step 0 obs)
  - both agents
  - per-turn (state_summary, action) for winner AND loser
  - per-game outcome (winner, ship counts, steps)

Output:
  data/round_robin_games.jsonl   — one line per game

This is the dataset that will train V(state) and inform map-conditional
strategy selection.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
import itertools
from collections import Counter
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def ref(name):
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    p = OPP_DIR / f"{name}.py"
    if p.exists():
        return str(p)
    p2 = ROOT / "agents" / f"{name}.py"
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(name)


def extract_map_features(obs):
    """Map features from step-0 observation. These are properties of the
    SEED — they don't change during the game. Used for map clustering."""
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    ang_vel = obs.get("angular_velocity", 0.0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0.0)

    homes = [p for p in planets if p[1] in (0, 1, 2, 3)]
    neutrals = [p for p in planets if p[1] == -1]

    # Production distribution
    prod_counter = Counter(p[6] for p in neutrals)
    prods = sorted(prod_counter.items())

    # Rotation classification
    rotating = sum(1 for p in planets if math.hypot(p[2]-50, p[3]-50) + p[4] < 50)

    # Distance metrics
    dist_home_to_nearest_neutral = float("inf")
    if homes and neutrals:
        h = homes[0]
        dist_home_to_nearest_neutral = min(
            math.hypot(h[2]-n[2], h[3]-n[3]) for n in neutrals)

    # Average neutral garrison
    avg_neutral_ships = sum(p[5] for p in neutrals) / max(1, len(neutrals))
    max_neutral_ships = max((p[5] for p in neutrals), default=0)

    # Production diversity (entropy)
    total_neutrals = sum(prod_counter.values())
    entropy = 0
    for c in prod_counter.values():
        p = c / total_neutrals
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
        "dist_home_to_nearest_neutral": round(dist_home_to_nearest_neutral, 2),
        "avg_neutral_ships": round(avg_neutral_ships, 1),
        "max_neutral_ships": max_neutral_ships,
        "prod_entropy": round(entropy, 3),
    }


def summarize_state_for_winner(obs, player):
    """Compact per-turn state. Just what we need for training V(state)."""
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)

    my_ships = sum(p[5] for p in planets if p[1] == player) + \
               sum(f[6] for f in fleets if f[1] == player)
    opp_ships = sum(p[5] for p in planets if p[1] != player and p[1] != -1) + \
                sum(f[6] for f in fleets if f[1] != player)
    my_planets = sum(1 for p in planets if p[1] == player)
    opp_planets = sum(1 for p in planets if p[1] != player and p[1] != -1)
    neutral_planets = sum(1 for p in planets if p[1] == -1)
    my_prod = sum(p[6] for p in planets if p[1] == player)
    opp_prod = sum(p[6] for p in planets if p[1] != player and p[1] != -1)
    my_fleet_ships = sum(f[6] for f in fleets if f[1] == player)
    opp_fleet_ships = sum(f[6] for f in fleets if f[1] != player)

    # Centrality (avg distance to center of mass for owned planets)
    if my_planets > 0:
        own_planets = [p for p in planets if p[1] == player]
        cx = sum(p[2] for p in own_planets) / my_planets
        cy = sum(p[3] for p in own_planets) / my_planets
        my_centrality = math.hypot(cx-50, cy-50)
    else:
        my_centrality = 0

    return {
        "step": step,
        "my_ships": my_ships,
        "opp_ships": opp_ships,
        "my_planets": my_planets,
        "opp_planets": opp_planets,
        "neutral_planets": neutral_planets,
        "my_prod": my_prod,
        "opp_prod": opp_prod,
        "my_fleet_ships": my_fleet_ships,
        "opp_fleet_ships": opp_fleet_ships,
        "my_centrality": round(my_centrality, 2),
    }


def run_game(agents, seed):
    """Run one game, return full per-player traces + map features + outcome."""
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
        return {"draw": True, "agents": agents, "seed": seed,
                "steps": len(env.steps), "rewards": rewards}, None

    # Map features from step 0
    map_features = extract_map_features(env.steps[0][0].observation)

    # Per-player traces (only state summaries, no full obs to keep file small)
    traces = {}
    for p_idx in range(len(agents)):
        traj = []
        for step_idx, step_data in enumerate(env.steps[:-1]):
            entry = step_data[p_idx]
            obs = entry.observation
            action = entry.action or []
            traj.append({
                "state": summarize_state_for_winner(obs, p_idx),
                "action": action,
            })
        traces[p_idx] = traj

    return {
        "agents": agents,
        "seed": seed,
        "winner": winners[0],
        "num_players": len(agents),
        "steps": len(env.steps),
        "rewards": rewards,
        "map_features": map_features,
        "traces": traces,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="Number of seeds per pairing")
    ap.add_argument("--agents", nargs="+", default=None,
                    help="Subset of agents to include")
    ap.add_argument("--mode", choices=["2p", "4p", "both"], default="2p")
    ap.add_argument("--out", default="round_robin_games.jsonl")
    args = ap.parse_args()

    all_agents = ["lb1224", "peak1103", "mlhybrid", "search_lvf", "tactical2026",
                  "orbitbotnext", "structured", "proto1000", "reinforce958",
                  "sundodge", "main"]
    agents = args.agents or all_agents

    out_file = DATA_DIR / args.out
    if out_file.exists():
        out_file.unlink()

    pairings = []
    if args.mode in ("2p", "both"):
        for a, b in itertools.permutations(agents, 2):
            pairings.append(("2p", [a, b]))
    if args.mode in ("4p", "both"):
        # Sample 4P lineups: pairs of (A,A,B,B), all-different chunks
        import random
        rng = random.Random(0)
        for _ in range(min(len(agents) * 3, 60)):
            lineup = rng.sample(agents, 4) if len(agents) >= 4 else None
            if lineup:
                pairings.append(("4p", lineup))

    total = len(pairings) * args.seeds
    print(f"Plan: {len(pairings)} pairings × {args.seeds} seeds = {total} games")
    print(f"Agents: {agents}")
    t0 = time.time()
    saved = errors = draws = 0
    f_out = open(out_file, "a")
    for seed in range(args.seeds):
        for mode, lineup in pairings:
            res, err = run_game(lineup, seed)
            if err:
                errors += 1
                continue
            if res.get("draw"):
                draws += 1
                continue
            f_out.write(json.dumps(res) + "\n")
            f_out.flush()
            saved += 1
        elapsed = time.time() - t0
        rate = (seed + 1) / args.seeds
        eta = elapsed / max(rate, 0.001) - elapsed
        print(f"[seed {seed+1}/{args.seeds}] saved={saved} err={errors} "
              f"draw={draws} elapsed={elapsed:.0f}s eta={eta:.0f}s",
              flush=True)
    f_out.close()
    print(f"\nDone: {saved} games saved to {out_file}")


if __name__ == "__main__":
    main()
