"""Run one 4P game with `agent` as player `pos`. After each step, compute
rank/gap_above/gap_below from raw state and print, so we can check whether
v101's stable_second predicate ever fires in practice.

Usage: .venv/bin/python scripts/diag_rank.py --seed 0 --pos 0
"""
import argparse, logging, os, sys
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agents"))

# Import by module name (not `agents.v101_rank_aware`) — the `agents` package
# name collides with kaggle_environments.envs.lux_ai_s3.agents.
from v101_rank_aware import _estimate_rank  # noqa: E402
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet


def ref(name):
    if name == "main":
        return str(ROOT / "main.py")
    p = ROOT / "agents" / f"{name}.py"
    if p.exists():
        return str(p)
    p = ROOT / "opponents" / f"{name}.py"
    if p.exists():
        return str(p)
    raise FileNotFoundError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos", type=int, default=0)
    ap.add_argument("--agent", default="v98_horz2p")
    ap.add_argument("--opps", nargs=3,
                    default=["mlhybrid", "structured", "proto1000"])
    args = ap.parse_args()

    agents = [ref(o) for o in args.opps]
    agents.insert(args.pos, ref(args.agent))
    env = make("orbit_wars", configuration={"seed": args.seed}, debug=False)
    env.run(agents)

    player = args.pos
    stable_count = 0
    rank_hist = []
    print(f"seed={args.seed} pos={player} agent={args.agent} opps={args.opps}")
    print(f"{'step':>4} {'rank':>4} {'g_abv':>7} {'g_blw':>7} {'stable2':>8}")
    for step_states in env.steps:
        obs = step_states[player].observation
        raw_planets = obs.get("planets", []) or []
        raw_fleets = obs.get("fleets", []) or []
        step_now = obs.get("step", 0) or 0
        all_owners = {p[1] for p in raw_planets if p[1] >= 0}
        for f in raw_fleets:
            if f[1] >= 0:
                all_owners.add(f[1])
        all_owners.add(player)
        max_player = max(all_owners) if all_owners else player
        num_players = 4 if max_player >= 2 else 2
        if num_players != 4:
            continue
        planets = [Planet(*p) for p in raw_planets]
        fleets = [Fleet(*f) for f in raw_fleets]
        rank, ga, gb = _estimate_rank(planets, fleets, player, num_players)
        stable = step_now >= 40 and rank == 2 and gb >= 0.08 and ga >= 0.06
        if stable:
            stable_count += 1
        rank_hist.append(rank)
        if step_now % 10 == 0 or stable:
            ga_s = f"{ga:.3f}" if ga != float('inf') else "  inf"
            gb_s = f"{gb:.3f}" if gb != float('inf') else "  inf"
            print(f"{step_now:>4} {rank:>4} {ga_s:>7} {gb_s:>7} "
                  f"{'YES' if stable else '':>8}")
    print(f"\nstable_second active for {stable_count} steps (out of {len(rank_hist)})")
    from collections import Counter
    c = Counter(rank_hist)
    print(f"rank distribution: {dict(c)}")
    final_reward = env.steps[-1][player].reward
    print(f"final reward = {final_reward}")


if __name__ == "__main__":
    main()
