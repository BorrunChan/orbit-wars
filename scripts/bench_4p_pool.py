"""
Bench main in 4P games where the 3 opps are randomly sampled from a pool.

Each seed plays main at all 4 positions × sample 3 opps randomly. Tracks
per-position win rate (to verify the slot-2/3 fix worked).

Usage:
  .venv/bin/python scripts/bench_4p_pool.py --seeds 10 \\
        --pool mlhybrid lb1224 structured proto1000 orbitbotnext
"""
import argparse, logging, os, random, sys, time
from collections import defaultdict
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"


AGENTS_DIR = ROOT / "agents"


def ref(name):
    if name == "main":
        return str(ROOT / "main.py")
    for d in (AGENTS_DIR, OPP_DIR):
        p = d / f"{name}.py"
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="main")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--pool", nargs="+",
                    default=["mlhybrid", "lb1224", "structured", "proto1000",
                              "orbitbotnext"])
    ap.add_argument("--rotate-pos", action="store_true", default=True)
    args = ap.parse_args()

    rng = random.Random(0)
    print(f"4P bench: {args.agent} in pool of {args.pool}, {args.seeds} seeds × 4 pos")
    t0 = time.time()
    wins_by_pos = defaultdict(lambda: [0, 0])  # [wins, total]
    wins_total = 0; games_total = 0
    for seed in range(args.seeds):
        for pos in range(4):
            opps = rng.sample(args.pool, 3)
            agents = [ref(o) for o in opps]
            agents.insert(pos, ref(args.agent))
            env = make("orbit_wars", configuration={"seed": seed},
                       debug=False)
            try:
                env.run(agents)
            except Exception as e:
                print(f"  seed={seed} pos={pos} CRASH: {e}")
                continue
            r = env.steps[-1][pos].reward
            won = r is not None and r > 0
            wins_by_pos[pos][0] += int(won); wins_by_pos[pos][1] += 1
            wins_total += int(won); games_total += 1
            color = "\033[32m" if won else "\033[31m"
            print(f"  seed={seed} pos={pos}  vs {','.join(opps):<40} "
                  f"{color}{'WIN' if won else 'LOSS'}\033[0m", flush=True)
    elapsed = time.time() - t0
    print()
    print(f"{'pos':>4} {'wins':>5} {'games':>5} {'WR':>6}")
    for pos in range(4):
        w, n = wins_by_pos[pos]
        wr = w / max(1, n) * 100
        flag = "" if wr >= 25 else "  ⚠️"
        print(f"  {pos:>2} {w:>5} {n:>5} {wr:>5.0f}%{flag}")
    print(f"\nTotal: {wins_total}/{games_total} = "
          f"{wins_total/max(1,games_total)*100:.0f}%  time={elapsed:.0f}s")


if __name__ == "__main__":
    main()
