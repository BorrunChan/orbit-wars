"""
2P head-to-head bench: each variant vs each opponent at both positions for
N seeds. Reports per-opponent and aggregate WR.

Usage:
  .venv/bin/python scripts/bench_2p_pool.py --variants main v125_coop_attack \
    --pool mlhybrid lb1224 structured proto1000 orbitbotnext --seeds 5
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
    ap.add_argument("--pool", nargs="+",
                    default=["mlhybrid", "lb1224", "structured", "proto1000",
                              "orbitbotnext"])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    print(f"Variants: {args.variants}")
    print(f"Pool: {args.pool}")
    print(f"Seeds: {args.seeds} × 2 positions × {len(args.pool)} opps "
          f"= {args.seeds * 2 * len(args.pool)} games/variant\n", flush=True)

    # variant -> opp -> [wins, total]
    results = {v: defaultdict(lambda: [0, 0]) for v in args.variants}
    t0 = time.time()

    for var in args.variants:
        print(f"\n=== {var} ===", flush=True)
        var_t0 = time.time()
        for opp in args.pool:
            opp_path = ref(opp)
            var_path = ref(var)
            for seed in range(args.seeds):
                for pos in range(2):
                    agents = [opp_path, opp_path]
                    agents[pos] = var_path
                    env = make("orbit_wars",
                               configuration={"seed": seed}, debug=False)
                    try:
                        env.run(agents)
                    except Exception as e:
                        print(f"  vs {opp} seed={seed} pos={pos} CRASH: {e}",
                              flush=True)
                        results[var][opp][1] += 1
                        continue
                    r_var = env.steps[-1][pos].reward
                    r_opp = env.steps[-1][1 - pos].reward
                    won = r_var is not None and r_opp is not None and r_var > r_opp
                    results[var][opp][0] += int(won)
                    results[var][opp][1] += 1
                    tag = "W" if won else "L"
                    print(f"  vs {opp:<14} seed={seed} pos={pos}  "
                          f"{tag} (r={r_var},{r_opp})", flush=True)
        elapsed = time.time() - var_t0
        print(f"  {var} took {elapsed:.0f}s", flush=True)

    # Summary
    print("\n" + "=" * 78)
    header = f"{'variant':<24}"
    for opp in args.pool:
        header += f" {opp[:10]:>10}"
    header += f" {'total':>10}"
    print(header)
    print("-" * 78)
    for var in args.variants:
        row = f"{var:<24}"
        tw = tg = 0
        for opp in args.pool:
            w, g = results[var][opp]
            tw += w; tg += g
            pct = 100 * w / g if g else 0
            row += f" {w}/{g} {pct:>3.0f}%".rjust(11)
        total_pct = 100 * tw / tg if tg else 0
        row += f" {tw}/{tg} {total_pct:>3.0f}%".rjust(11)
        print(row)
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
