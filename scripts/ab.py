"""
Fast A/B benchmark: main.py vs one or more strong opponents.

Usage:
    .venv/bin/python scripts/ab.py [agent] [seeds] [opps...]
    .venv/bin/python scripts/ab.py main 5 lb1224 proto1000 structured
"""

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


def ref(name):
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    for d in (AGENTS_DIR, OPP_DIR):
        p = d / f"{name}.py"
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def main():
    agent = sys.argv[1] if len(sys.argv) > 1 else "main"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    opps = sys.argv[3:] if len(sys.argv) > 3 else ["lb1224", "proto1000", "structured"]
    print(f"\n{agent} ({ref(agent)}) vs {opps}, {seeds} seeds each side")
    t0 = time.time()
    total_w = total_g = 0
    for opp in opps:
        w = l = 0
        for s in range(seeds):
            for swap in (False, True):
                a, b = (ref(opp), ref(agent)) if swap else (ref(agent), ref(opp))
                env = make("orbit_wars", configuration={"seed": s}, debug=False)
                env.run([a, b])
                me_idx = 1 if swap else 0
                r = env.steps[-1][me_idx].reward
                if r > 0:
                    w += 1
                else:
                    l += 1
        g = w + l
        total_w += w; total_g += g
        wr = w / g
        color = "\033[32m" if wr >= 0.4 else "\033[31m" if wr <= 0.2 else "\033[33m"
        print(f"  vs {opp:<13} {color}{w:>2}W {l:>2}L  WR={wr:.0%}\033[0m")
    print(f"Total: {total_w}/{total_g} = {total_w/total_g:.0%}  time={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
