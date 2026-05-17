"""Bench an agent vs the 3 swarm tiers (light/medium/extreme) in 2P + 4P.

For 2P: head-to-head v98 vs swarm_X.
For 4P: v98 at all 4 positions × 3 copies of swarm_X filling the rest.

Reports per-tier per-format win rate + observed launches/game by the swarm bot
(so we can verify each tier hits its intended intensity).

Usage:
  .venv/bin/python scripts/bench_swarm_tiers.py --agent main --seeds 3
"""
import argparse, logging, os, sys, time
from collections import defaultdict
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
AGENTS_DIR = ROOT / "agents"
OPP_DIR = ROOT / "opponents"

TIERS = ["swarm_light", "swarm_medium", "swarm_extreme"]


def ref(name):
    if name == "main":
        return str(ROOT / "main.py")
    for d in (AGENTS_DIR, OPP_DIR):
        p = d / f"{name}.py"
        if p.exists():
            return str(p)
    raise FileNotFoundError(name)


def count_opp_launches(env, opp_slots):
    """Sum len(actions) for each opponent slot across all steps."""
    total = 0
    for step_data in env.steps[:-1]:
        for slot in opp_slots:
            if slot < len(step_data):
                action = step_data[slot].action or []
                if isinstance(action, list):
                    total += len(action)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="main")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--tiers", nargs="+", default=TIERS)
    ap.add_argument("--formats", nargs="+", default=["2P", "4P"])
    args = ap.parse_args()

    agent_path = ref(args.agent)
    print(f"Agent: {args.agent} ({agent_path})")
    print(f"Seeds: {args.seeds}  Tiers: {args.tiers}  Formats: {args.formats}\n")

    t0 = time.time()
    results = defaultdict(lambda: [0, 0, 0, 0])

    for tier in args.tiers:
        opp_path = ref(tier)
        for fmt in args.formats:
            n_players = 2 if fmt == "2P" else 4
            for seed in range(args.seeds):
                for pos in range(n_players):
                    agents = [opp_path] * n_players
                    agents[pos] = agent_path
                    opp_slots = [i for i in range(n_players) if i != pos]

                    env = make("orbit_wars", configuration={"seed": seed},
                               debug=False)
                    try:
                        env.run(agents)
                    except Exception as e:
                        print(f"  {tier} {fmt} seed={seed} pos={pos} CRASH: {e}",
                              flush=True)
                        results[(tier, fmt)][1] += 1
                        continue
                    r = env.steps[-1][pos].reward
                    won = r is not None and r > 0
                    n_launches = count_opp_launches(env, opp_slots)
                    n_steps = len(env.steps) - 1
                    avg_launches_per_opp = (n_launches / max(1, len(opp_slots))
                                            if opp_slots else 0)
                    tag = "W" if won else "L"
                    print(f"  {tier:<14} {fmt} seed={seed} pos={pos}  {tag}  "
                          f"opp_launches/bot={avg_launches_per_opp:.0f}  "
                          f"steps={n_steps}", flush=True)
                    results[(tier, fmt)][0] += int(won)
                    results[(tier, fmt)][1] += 1
                    results[(tier, fmt)][2] += avg_launches_per_opp
                    results[(tier, fmt)][3] += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 72)
    print(f"{'tier':<16} {'fmt':>4} {'wins':>5} {'games':>6} "
          f"{'WR':>6} {'avg_launches/opp':>18}")
    print("-" * 72)
    for tier in args.tiers:
        for fmt in args.formats:
            w, n, sl, sg = results[(tier, fmt)]
            wr = w / max(1, n) * 100
            avg_l = sl / max(1, sg)
            print(f"{tier:<16} {fmt:>4} {w:>5} {n:>6} {wr:>5.0f}% "
                  f"{avg_l:>18.0f}")
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
