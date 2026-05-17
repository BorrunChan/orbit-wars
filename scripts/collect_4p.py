"""
4P self-play data collector for V(state) training.

Records per-turn state summary (from each player's POV) + final winner.
Mixes lineups to get diverse state distributions:
  - 4×main (pure self-play)
  - 3×main + 1×strong
  - 2×main + 2×strong (1 of each)
  - main + 3 strong (mlhybrid, lb1224, structured)
  - rotated positions

Output: data/v4p_games.jsonl
  one line per game with: {agents, seed, winner, num_players=4,
                            traces: {player_idx: [{state, action}, ...]}}

Usage:
  .venv/bin/python -u scripts/collect_4p.py [seeds=50]
"""

import argparse, json, logging, math, os, sys, time, random
from collections import Counter
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"
DATA = ROOT / "data" / "v4p_games.jsonl"


def ref(name):
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    return str(OPP_DIR / f"{name}.py")


def compact_state(obs, player):
    """Compact per-turn state — focuses on features useful for V."""
    if isinstance(obs, dict):
        g = obs.get
    else:
        g = lambda k, d=None: getattr(obs, k, d)
    planets = g("planets", []) or []
    fleets = g("fleets", []) or []
    step = g("step", 0) or 0
    n_players = 4  # this collector is 4P-only

    # Counts per owner
    owner_planets = [0, 0, 0, 0]
    owner_ships = [0, 0, 0, 0]
    owner_prod = [0, 0, 0, 0]
    neutral_planets = 0
    neutral_ships = 0
    for p in planets:
        if p[1] == -1:
            neutral_planets += 1
            neutral_ships += p[5]
        elif 0 <= p[1] < 4:
            owner_planets[p[1]] += 1
            owner_ships[p[1]] += p[5]
            owner_prod[p[1]] += p[6]
    owner_fleet_ships = [0, 0, 0, 0]
    for f in fleets:
        if 0 <= f[1] < 4:
            owner_fleet_ships[f[1]] += f[6]

    # Centrality (avg distance to center of mass for my planets)
    my_pl = [p for p in planets if p[1] == player]
    if my_pl:
        cx = sum(p[2] for p in my_pl) / len(my_pl)
        cy = sum(p[3] for p in my_pl) / len(my_pl)
        my_centrality = math.hypot(cx - 50, cy - 50)
    else:
        my_centrality = 0

    # Min distance from my planets to any other owner's planets
    min_enemy_dist = 100.0
    enemy_pl = [p for p in planets if p[1] != -1 and p[1] != player]
    if my_pl and enemy_pl:
        min_enemy_dist = min(
            math.hypot(m[2] - e[2], m[3] - e[3])
            for m in my_pl for e in enemy_pl
        )

    return {
        "step": step,
        "my": {
            "ships": owner_ships[player] + owner_fleet_ships[player],
            "planets": owner_planets[player],
            "prod": owner_prod[player],
            "fleet_ships": owner_fleet_ships[player],
            "centrality": round(my_centrality, 2),
        },
        # All opps (the 3 others) — also encode aggregate
        "opps": [
            {
                "ships": owner_ships[i] + owner_fleet_ships[i],
                "planets": owner_planets[i],
                "prod": owner_prod[i],
                "fleet_ships": owner_fleet_ships[i],
            }
            for i in range(4) if i != player
        ],
        "neutral_planets": neutral_planets,
        "neutral_ships": neutral_ships,
        "min_enemy_dist": round(min_enemy_dist, 1),
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
        return None, "draw"
    winner = winners[0]
    # Per-player traces (state every turn)
    traces = {}
    for p_idx in range(len(agents)):
        traj = []
        for step_idx, step_data in enumerate(env.steps[:-1]):
            entry = step_data[p_idx]
            obs = entry.observation
            action = entry.action or []
            traj.append({
                "state": compact_state(obs, p_idx),
                "action": action,
            })
        traces[p_idx] = traj
    return {
        "agents": agents,
        "seed": seed,
        "winner": winner,
        "winner_agent": agents[winner],
        "num_players": 4,
        "steps": len(env.steps),
        "rewards": rewards,
        "traces": traces,
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Start seed at this offset (e.g., 50 to continue past first batch)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing data file instead of overwriting")
    args = ap.parse_args()

    if not args.append and DATA.exists():
        DATA.unlink()

    rng = random.Random(0)
    # Define lineup templates — sampled per game to get diversity
    strong_pool = ["lb1224", "mlhybrid", "structured", "proto1000",
                   "orbitbotnext", "sundodge", "peak1103"]

    def sample_lineup():
        r = rng.random()
        if r < 0.20:  # pure self-play
            return ["main"] * 4
        elif r < 0.35:  # 3 main + 1 strong
            opp = rng.choice(strong_pool)
            base = ["main"] * 3 + [opp]
            rng.shuffle(base)
            return base
        elif r < 0.55:  # 2 main + 2 strong (1 each)
            opps = rng.sample(strong_pool, 2)
            base = ["main"] * 2 + opps
            rng.shuffle(base)
            return base
        elif r < 0.80:  # 1 main + 3 strong
            opps = rng.sample(strong_pool, 3)
            base = ["main"] + opps
            rng.shuffle(base)
            return base
        else:  # 4 strong (no main)
            base = rng.sample(strong_pool, 4)
            return base

    # 1 lineup per seed, but multiple lineups per seed for data efficiency
    games_per_seed = 4
    total = args.seeds * games_per_seed
    print(f"Plan: {args.seeds} seeds × {games_per_seed} lineups = {total} games")

    t0 = time.time()
    saved = err = 0
    f_out = open(DATA, "a")
    for seed_i in range(args.seeds):
        seed = seed_i + args.seed_offset
        for _ in range(games_per_seed):
            lineup = sample_lineup()
            res, e = run_game(lineup, seed)
            if e:
                err += 1
                continue
            f_out.write(json.dumps(res) + "\n")
            f_out.flush()
            saved += 1
        elapsed = time.time() - t0
        eta = elapsed / max(0.001, (seed_i + 1) / args.seeds) - elapsed
        print(f"[seed {seed_i+1}/{args.seeds} (id={seed})] saved={saved} err={err} "
              f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    f_out.close()
    print(f"\nDone: {saved} games at {DATA}")


if __name__ == "__main__":
    main()
