#!/usr/bin/env python
"""Smoke test for new agent versions.

Runs the candidate agent vs opponents/structured.py for 3 seeds.
Fails if the candidate wins zero games — this catches catastrophic
regressions like v117 (0% bench rate from unguided pressure shots).

Usage:
    python scripts/smoke_test.py agents/v117_pressure.py
    python scripts/smoke_test.py main.py

Exit code:
    0  PASS  (won at least 1/3 games)
    1  FAIL  (lost all 3 games — likely broken strategy)
    2  ERROR (exception during run — likely broken code)

This is intentionally cheap (~30-60s) to be safe as a pre-commit hook.
For full bench, use scripts/bench_variants.py.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OPPONENT = str(REPO_ROOT / "opponents" / "structured.py")
SEEDS = [1, 7, 42]


def run_one(agent_path: str, seed: int) -> tuple[bool, str]:
    """Return (won, status_str)."""
    try:
        from kaggle_environments import make
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([agent_path, OPPONENT])
        final = env.steps[-1]
        won = final[0].reward > final[1].reward
        steps = len(env.steps)
        return won, f"seed={seed} steps={steps} reward={final[0].reward:+}"
    except Exception as e:
        return False, f"seed={seed} EXCEPTION: {type(e).__name__}: {e}"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <agent.py>", file=sys.stderr)
        return 2

    agent = sys.argv[1]
    if not os.path.exists(agent):
        print(f"❌ ERROR: agent file not found: {agent}", file=sys.stderr)
        return 2
    if not os.path.exists(OPPONENT):
        print(f"❌ ERROR: opponent not found: {OPPONENT}", file=sys.stderr)
        return 2

    print(f"smoke_test: {agent} vs {Path(OPPONENT).name} on seeds {SEEDS}")
    t0 = time.time()
    wins = 0
    error = False
    for seed in SEEDS:
        won, status = run_one(agent, seed)
        if "EXCEPTION" in status:
            error = True
            print(f"  ✗ {status}")
        elif won:
            wins += 1
            print(f"  ✓ {status}")
        else:
            print(f"  ✗ {status}")

    elapsed = time.time() - t0
    print(f"smoke_test: {wins}/{len(SEEDS)} wins in {elapsed:.1f}s")

    if error:
        print(f"❌ ERROR: agent crashed", file=sys.stderr)
        return 2
    if wins == 0:
        print(
            f"❌ FAIL: {agent} lost all {len(SEEDS)} games vs structured. "
            f"Do not deploy.",
            file=sys.stderr,
        )
        return 1
    print(f"✅ PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
