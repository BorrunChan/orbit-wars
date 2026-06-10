"""League self-play orchestrator: iterate gen -> train -> eval, growing a pool
of value-agent snapshots so the value never overfits one opponent (anti-v82).

Each iteration:
  1. generate games among (producer variants + external opps + past snapshots)
  2. train value on ALL data so far
  3. eval: value-agent(latest) vs exp48 head-to-head; print win rate (local gate)
  4. snapshot weights into the pool for next iteration

Run on the 3060 box:
    python rl/run_league.py --iters 6 --games_per_iter 300 --workers 6

Stop when head-to-head win rate vs exp48 stops improving. Then export + submit
the best snapshot (see build_submission.py) and validate on Kaggle (~15 games).
"""
from __future__ import annotations
import argparse, os, subprocess, sys, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
DATA = os.path.join("rl", "data")


def sh(cmd):
    print("  $", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def eval_vs_exp48(weights, n_games, workers):
    """Head-to-head value-agent(weights) vs exp48, both positions. Returns win%."""
    import importlib.util, logging, random
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    os.environ["VALUE_WEIGHTS"] = weights
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    from kaggle_environments import make
    wins = g = 0
    rng = random.Random(0)
    for k in range(n_games):
        seed = rng.randrange(1_000_000)
        nP = 4 if k % 2 == 0 else 2
        agents = ["single-size/main_exp48.py"] * (nP - 1)
        pos = rng.randrange(nP)
        agents.insert(pos, "rl/value_agent.py")
        try:
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run([os.path.join(ROOT, a) for a in agents])
        except Exception:
            continue
        rew = [env.state[i]["reward"] for i in range(nP)]
        if rew[pos] is None:
            continue
        mx = max(r for r in rew if r is not None)
        wins += 1 if (rew[pos] == mx and rew.count(mx) == 1) else 0
        g += 1
    return 100.0 * wins / max(1, g), g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=6)
    ap.add_argument("--games_per_iter", type=int, default=300)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--eval_games", type=int, default=40)
    args = ap.parse_args()
    os.makedirs(os.path.join(ROOT, DATA), exist_ok=True)

    all_data = []
    weights_latest = None
    for it in range(args.iters):
        data_path = f"{DATA}/iter{it}.jsonl"
        gen = [PY, "rl/gen_selfplay.py", "--games", str(args.games_per_iter),
               "--out", data_path, "--workers", str(args.workers)]
        if weights_latest:   # add current value agent + recent snapshots to the pool
            gen += ["--extra", "rl/value_agent.py", "--weights", weights_latest]
        sh(gen)
        all_data.append(data_path)

        w_path = f"{DATA}/value_iter{it}.json"
        sh([PY, "rl/train_value.py", "--data", *all_data, "--out", w_path])
        # publish as latest (value_agent + gen pick this up)
        import shutil
        shutil.copy(os.path.join(ROOT, w_path), os.path.join(ROOT, DATA, "value_latest.json"))
        weights_latest = os.path.join(DATA, "value_latest.json")

        wr, g = eval_vs_exp48(os.path.join(ROOT, weights_latest), args.eval_games, args.workers)
        print(f"\n=== iter {it}: value vs exp48 head-to-head = {wr:.0f}% over {g} games ===\n", flush=True)


if __name__ == "__main__":
    main()
