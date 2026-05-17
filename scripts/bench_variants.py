"""
Compare multiple agent variants in 4P. Runs each variant for `--seeds` seeds × 4 positions.

Usage:
  .venv/bin/python scripts/bench_variants.py --variants v43_4pfix v44_kovi v45_softerthresh v46_bigfleet --seeds 5
"""
import argparse, logging, os, random, sys, time
from collections import defaultdict
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "agents"
OPP_DIR = ROOT / "opponents"


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
    ap.add_argument("--variants", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--pool", nargs="+",
                    default=["mlhybrid", "lb1224", "structured", "proto1000",
                              "orbitbotnext"])
    args = ap.parse_args()

    print(f"Variants: {args.variants}")
    print(f"Pool: {args.pool}")
    print(f"Seeds: {args.seeds} × 4 positions = {args.seeds*4} games/variant\n")

    results = {}  # variant -> [wins_by_pos]
    t0 = time.time()
    for var in args.variants:
        print(f"\n=== {var} ===")
        rng = random.Random(0)  # same opp sampling per variant for fairness
        wins_by_pos = [[0, 0] for _ in range(4)]
        var_start = time.time()
        for seed in range(args.seeds):
            for pos in range(4):
                opps = rng.sample(args.pool, 3)
                agents = [ref(o) for o in opps]
                agents.insert(pos, ref(var))
                env = make("orbit_wars", configuration={"seed": seed},
                           debug=False)
                try:
                    env.run(agents)
                except Exception as e:
                    print(f"  seed={seed} pos={pos} CRASH: {e}", flush=True)
                    wins_by_pos[pos][1] += 1
                    continue
                r = env.steps[-1][pos].reward
                won = r is not None and r > 0
                wins_by_pos[pos][0] += int(won); wins_by_pos[pos][1] += 1
                tag = "W" if won else "L"
                print(f"  seed={seed} pos={pos} vs {','.join(opps):<40} {tag}",
                      flush=True)
        results[var] = wins_by_pos
        elapsed = time.time() - var_start
        print(f"  {var} took {elapsed:.0f}s")

    # Summary
    print("\n" + "=" * 70)
    print(f"{'variant':<22} {'pos0':>7} {'pos1':>7} {'pos2':>7} {'pos3':>7} "
          f"{'total':>7}")
    print("-" * 70)
    for var, wbp in results.items():
        cells = []
        tw = tg = 0
        for pos in range(4):
            w, n = wbp[pos]
            cells.append(f"{w}/{n}={w/max(1,n)*100:.0f}%")
            tw += w; tg += n
        cells.append(f"{tw}/{tg}={tw/max(1,tg)*100:.0f}%")
        print(f"  {var:<20}", end="")
        for c in cells:
            print(f"{c:>10}", end="")
        print()
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
