"""
Full benchmark across all game formats:
  - 2P vs strong opponents (lb1224, proto1000, structured)
  - 2P vs weak baselines (random, starter, v0, v1, v2d)
  - 4P with main + 3 random
  - 4P with main + 3 starter
  - 4P with main + mix of strong
  - 4P with main + mix of weak

Usage:
    .venv/bin/python scripts/full_bench.py [seeds]
"""
import logging, os, sys, time, itertools
from pathlib import Path
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make

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


def play_2p(a, b, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([ref(a), ref(b)])
    return env.steps[-1][0].reward, env.steps[-1][1].reward


def play_4p(agents, seed):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run([ref(a) for a in agents])
    return [s.reward for s in env.steps[-1]]


def bench_2p(opp_list, seeds, label):
    """Each opp: seeds each side."""
    print(f"\n=== {label} ===")
    total_w = total_g = 0
    for opp in opp_list:
        w = l = 0
        for s in range(seeds):
            for swap in (False, True):
                if swap:
                    r0, r1 = play_2p(opp, "main", s)
                    r = r1
                else:
                    r0, r1 = play_2p("main", opp, s)
                    r = r0
                if r > 0: w += 1
                else: l += 1
        g = w + l
        total_w += w; total_g += g
        wr = w / g
        color = "\033[32m" if wr >= 0.6 else "\033[31m" if wr <= 0.3 else "\033[33m"
        print(f"  vs {opp:<14} {color}{w:>2}W {l:>2}L  WR={wr:.0%}\033[0m")
    print(f"  {label} total: {total_w}/{total_g} = {total_w/total_g:.0%}")
    return total_w, total_g


def bench_4p(opp_list, seeds, label):
    """4P: main + 3 of opp_list, rotating positions across seeds."""
    print(f"\n=== {label} ===")
    w = l = 0
    for s in range(seeds):
        for pos in range(4):
            agents = list(opp_list)
            agents.insert(pos, "main")
            rewards = play_4p(agents, s)
            if rewards[pos] > 0: w += 1
            else: l += 1
    g = w + l
    wr = w / g
    color = "\033[32m" if wr >= 0.4 else "\033[31m" if wr <= 0.15 else "\033[33m"
    print(f"  vs {opp_list}  {color}{w}W {l}L  WR={wr:.0%}\033[0m")
    return w, g


def main():
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    t0 = time.time()
    summary = []

    # 2P vs strong
    w, g = bench_2p(["lb1224", "proto1000", "structured"], seeds, "2P vs STRONG")
    summary.append(("2P strong", w, g))

    # 2P vs weak
    w, g = bench_2p(["random", "starter", "v2d_conserve"], seeds, "2P vs WEAK")
    summary.append(("2P weak", w, g))

    # 4P: + 3 random
    w, g = bench_4p(["random", "random", "random"], seeds, "4P + 3 random")
    summary.append(("4P random", w, g))

    # 4P: + 3 starter
    w, g = bench_4p(["starter", "starter", "starter"], seeds, "4P + 3 starter")
    summary.append(("4P starter", w, g))

    # 4P: + 3 strong (one of each)
    w, g = bench_4p(["lb1224", "proto1000", "structured"], seeds, "4P + 3 STRONG")
    summary.append(("4P strong", w, g))

    # 4P: + 2 strong 1 weak
    w, g = bench_4p(["lb1224", "structured", "v2d_conserve"], seeds, "4P + 2strong 1mid")
    summary.append(("4P mixed", w, g))

    print(f"\n{'='*50}")
    print(f"{'Format':<25} {'WR':>10}")
    print(f"{'-'*50}")
    overall_w = overall_g = 0
    for name, w, g in summary:
        overall_w += w; overall_g += g
        print(f"{name:<25} {w:>3}/{g:<3}  ({w/g:.0%})")
    print(f"{'-'*50}")
    print(f"{'OVERALL':<25} {overall_w:>3}/{overall_g:<3}  ({overall_w/overall_g:.0%})")
    print(f"Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
