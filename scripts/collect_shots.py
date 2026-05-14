"""
Focused collector: v20 vs strong opps, save full per-turn state for shot
outcome extraction.

For each game:
  - state[player_idx][turn] = full planet list + fleet list (for ray-cast
    target identification and feature engineering)
  - actions per turn

Output:
  data/shot_games.jsonl

Usage:
  .venv/bin/python -u scripts/collect_shots.py [seeds]
"""
import json, logging, os, sys, time
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"
DATA = ROOT / "data" / "shot_games.jsonl"


def ref(name):
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    return str(OPP_DIR / f"{name}.py")


def compact_state(obs, player):
    planets = obs.get("planets", []) if isinstance(obs, dict) else getattr(obs, "planets", [])
    fleets = obs.get("fleets", []) if isinstance(obs, dict) else getattr(obs, "fleets", [])
    step = obs.get("step", 0) if isinstance(obs, dict) else getattr(obs, "step", 0)
    angvel = obs.get("angular_velocity", 0) if isinstance(obs, dict) else getattr(obs, "angular_velocity", 0)
    # Compact: per planet = [id, owner, x, y, ships, prod]; per fleet = [id, owner, x, y, angle, ships]
    pl = [[p[0], p[1], round(p[2], 2), round(p[3], 2), p[5], p[6]] for p in planets]
    fl = [[f[0], f[1], round(f[2], 2), round(f[3], 2), round(f[4], 4), f[6]] for f in fleets]
    return {"step": step, "ang_vel": angvel, "p": pl, "f": fl}


def run_game(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run([ref(a) for a in agents])
    except Exception as e:
        return None, str(e)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    if max(rewards) <= 0:
        return None, "draw"
    winner = rewards.index(max(rewards))
    # Save per-player full traces (state + action)
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
        "agents": agents, "seed": seed, "winner": winner,
        "num_players": len(agents), "rewards": rewards,
        "traces": traces,
    }, None


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    pairings = []
    # main vs each opponent (both sides), 3 seeds
    opps = ["mlhybrid", "orbitbotnext", "proto1000", "structured",
            "lb1224", "sundodge", "peak1103"]
    for opp in opps:
        pairings.append(["main", opp])
        pairings.append([opp, "main"])
    print(f"Plan: {len(pairings)} pairings × {seeds} seeds = "
          f"{len(pairings)*seeds} games")
    if DATA.exists():
        DATA.unlink()
    t0 = time.time()
    saved = err = 0
    with open(DATA, "w") as f:
        for s in range(seeds):
            for lineup in pairings:
                res, e = run_game(lineup, s)
                if e:
                    err += 1
                    continue
                f.write(json.dumps(res) + "\n")
                f.flush()
                saved += 1
            elapsed = time.time() - t0
            eta = elapsed / max(0.001, (s+1)/seeds) - elapsed
            print(f"[seed {s+1}/{seeds}] saved={saved} err={err} "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)
    print(f"\nDone: {saved} games at {DATA}")


if __name__ == "__main__":
    main()
