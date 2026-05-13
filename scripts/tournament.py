"""
Tournament harness — run round-robin matches between agents across many seeds.

Usage:
    .venv/bin/python scripts/tournament.py                  # default battery
    .venv/bin/python scripts/tournament.py --seeds 20       # more seeds
    .venv/bin/python scripts/tournament.py --quick          # 5 seeds, top-2 agents only
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"

from kaggle_environments import make  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "agents"


def agent_ref(name: str) -> str:
    """Resolve agent name to file path or builtin name."""
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    for d in (AGENTS_DIR, ROOT / "opponents"):
        p = d / f"{name}.py"
        if p.exists():
            return str(p)
    p2 = Path(name)
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(f"Unknown agent: {name}")


def play_match(a: str, b: str, seed: int) -> tuple[int, int, int]:
    """Returns (reward_a, reward_b, steps)."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([agent_ref(a), agent_ref(b)])
    final = env.steps[-1]
    return final[0].reward, final[1].reward, len(env.steps)


def run_pair(a: str, b: str, num_seeds: int, swap_sides: bool = True) -> dict:
    wins_a = losses_a = draws = 0
    total_steps = 0
    for seed in range(num_seeds):
        # Play with a as P0
        r0, r1, steps = play_match(a, b, seed)
        total_steps += steps
        if r0 > r1:
            wins_a += 1
        elif r0 < r1:
            losses_a += 1
        else:
            draws += 1
        if swap_sides:
            # Play with a as P1
            r0, r1, steps = play_match(b, a, seed)
            total_steps += steps
            if r1 > r0:
                wins_a += 1
            elif r1 < r0:
                losses_a += 1
            else:
                draws += 1
    games = wins_a + losses_a + draws
    return {
        "a": a, "b": b, "games": games,
        "wins_a": wins_a, "losses_a": losses_a, "draws": draws,
        "winrate_a": wins_a / games if games else 0.0,
        "avg_steps": total_steps / games if games else 0,
    }


def fmt_row(r: dict) -> str:
    wr = r["winrate_a"]
    color = "\033[32m" if wr >= 0.6 else "\033[31m" if wr <= 0.4 else "\033[33m"
    return (f"  {r['a']:<22} vs {r['b']:<22}  "
            f"{color}{r['wins_a']:>3}W {r['losses_a']:>3}L {r['draws']:>3}D"
            f"  WR={wr:.0%}\033[0m  avg_steps={r['avg_steps']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--agents", nargs="*", default=None,
                    help="Agents to include. Default: latest main vs full battery.")
    ap.add_argument("--quick", action="store_true",
                    help="Fewer seeds, smaller battery")
    args = ap.parse_args()

    if args.quick:
        seeds = 5
        battery = ["random", "starter", "v0_nearest"]
    else:
        seeds = args.seeds
        battery = args.agents or ["random", "starter", "v0_nearest"]

    challenger = "main"
    print(f"\n=== Tournament: '{challenger}' vs battery, {seeds} seeds (each side) ===\n")

    t0 = time.time()
    results = []
    for opp in battery:
        r = run_pair(challenger, opp, seeds)
        results.append(r)
        print(fmt_row(r))
    dt = time.time() - t0

    print(f"\nTotal time: {dt:.1f}s")
    overall_wins = sum(r["wins_a"] for r in results)
    overall_games = sum(r["games"] for r in results)
    print(f"Overall: {overall_wins}/{overall_games} = {overall_wins/overall_games:.0%}")


if __name__ == "__main__":
    main()
