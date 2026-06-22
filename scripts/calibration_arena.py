"""
Calibrated local test arena.

The problem this whole project hit: the naive local bench (win-rate vs a few
synthetic bots) is ANTI-correlated with Kaggle. This arena fixes that by using
our OWN past submissions as calibration anchors — each has a local agent file
AND a known real Kaggle settled score. If a local round-robin reproduces the
anchors' Kaggle ORDERING, the arena is predictive and we can read a new agent's
implied Kaggle rating from where it lands among the anchors.

Method:
  1. Round-robin the anchors (+ any test agents) in 2P, N seeds x both positions.
  2. Score each agent by win-rate in the round-robin.
  3. Spearman-correlate anchor local-winrate vs known Kaggle score. High rho =>
     arena is predictive.
  4. Read test agents' local rank against the anchor Kaggle scale.

Usage:
  .venv/bin/python scripts/calibration_arena.py --seeds 3
  .venv/bin/python scripts/calibration_arena.py --seeds 5 --test v130_logistics
"""
import argparse
import itertools
import logging
import os
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AG = ROOT / "agents"

# anchor agent file (in agents/) -> known Kaggle settled publicScore
ANCHORS = {
    "v20_searched": 755.2,
    "v95_total_swarm": 764.0,
    "v124_early_swarm": 776.1,
    "v93_twotier": 776.3,
    "v84_adaptive": 783.5,
    "v123_swarmer_detect": 784.9,
    "v70_no4pV": 800.4,
    "v98_horz2p": 803.8,
}


# Diverse opponent pool (proxy for the real Kaggle field). Anchors are scored
# vs THIS pool, not vs each other (similar anchors just draw). The question is
# whether WR-vs-pool reproduces the anchors' Kaggle ordering.
OPP_DIR = ROOT / "opponents"
POOL = ["structured", "lb1224", "mlhybrid", "orbitbotnext", "proto1000",
        "reinforce958", "peak1103", "sundodge"]


def ref(name):
    for d in (AG, OPP_DIR):
        p = d / f"{name}.py"
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def play(a, b, seed):
    """Return 1 if a wins, 0 if b wins/draw."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([ref(a), ref(b)])
    r0, r1 = env.steps[-1][0].reward, env.steps[-1][1].reward
    return 1 if (r0 is not None and r1 is not None and r0 > r1) else 0


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1)) if n > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--test", nargs="*", default=[])
    args = ap.parse_args()

    agents = list(ANCHORS.keys()) + args.test
    wins = {a: 0 for a in agents}
    games = {a: 0 for a in agents}

    print(f"agents: {len(agents)}  pool: {len(POOL)}  "
          f"games: {len(agents) * len(POOL) * args.seeds * 2}\n")

    for a in agents:
        for opp in POOL:
            for seed in range(args.seeds):
                for x, y, who in ((a, opp, 'a'), (opp, a, 'b')):
                    w = play(x, y, seed)
                    games[a] += 1
                    wins[a] += w if who == 'a' else (1 - w)
        print(f"  done {a}: {wins[a]}/{games[a]}", flush=True)

    wr = {a: 100 * wins[a] / games[a] for a in agents}

    # correlation on anchors only
    anchor_names = list(ANCHORS.keys())
    local = [wr[a] for a in anchor_names]
    kaggle = [ANCHORS[a] for a in anchor_names]
    rho = spearman(local, kaggle)

    print("\n" + "=" * 60)
    print(f"{'agent':<22}{'local_wr':>10}{'kaggle':>10}")
    print("-" * 60)
    for a in sorted(agents, key=lambda a: wr[a], reverse=True):
        kg = ANCHORS.get(a)
        kgs = f"{kg:.1f}" if kg is not None else "TEST/?"
        print(f"{a:<22}{wr[a]:>9.1f}%{kgs:>10}")
    print("-" * 60)
    print(f"Spearman rho (anchor local_wr vs Kaggle): {rho:+.3f}")
    print("  >0.6 => arena PREDICTIVE; <0.3 => not better than naive bench;")
    print("  <0   => anti-correlated (the problem we're trying to escape)")


if __name__ == "__main__":
    main()
