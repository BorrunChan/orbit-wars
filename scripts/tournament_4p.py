"""
4-player tournament. main agent + 3 opponents per game. We rotate which
opponent occupies which slot across seeds to be fair.

Usage:
    .venv/bin/python scripts/tournament_4p.py --seeds 8 --opp lb1224 proto1000 structured
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"

from kaggle_environments import make  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "agents"
OPP_DIR = ROOT / "opponents"


def agent_ref(name: str) -> str:
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    for d in (AGENTS_DIR, OPP_DIR):
        p = d / f"{name}.py"
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def play_4p(my_pos, opps, seed):
    """my_pos in 0..3; opps is list of 3 opponent names."""
    agents = list(opps)
    agents.insert(my_pos, agent_ref("main"))
    for i, a in enumerate(agents):
        if a not in ("random", "starter") and "/" not in a:
            agents[i] = agent_ref(a)
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    rewards = [s.reward for s in env.steps[-1]]
    return rewards[my_pos], rewards, len(env.steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--opp", nargs="+", default=["lb1224", "proto1000", "structured"])
    args = ap.parse_args()

    print(f"\n=== 4P Tournament: main + 3 of {args.opp} ===\n")
    t0 = time.time()
    wins = losses = 0
    for seed in range(args.seeds):
        for my_pos in range(4):
            r, all_r, steps = play_4p(my_pos, args.opp, seed)
            outcome = "WIN " if r > 0 else "LOSS"
            color = "\033[32m" if r > 0 else "\033[31m"
            if r > 0:
                wins += 1
            else:
                losses += 1
            print(f"  seed={seed} pos={my_pos}  {color}{outcome}\033[0m  rewards={all_r}  steps={steps}")
    total = wins + losses
    print(f"\nTotal: {wins}W / {losses}L = {wins/total:.0%}  time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
