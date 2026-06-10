"""Generate league self-play games and collect (features, win) training data.

Plays games among a POOL of agent files (producer variants + external opponents
+ past value-agent snapshots). A diverse pool is the anti-overfit guard that v82
lacked (do NOT train on pure self-play vs one opponent). Records features for
EVERY player at sampled game-progress fractions, labeled by whether that player
finished rank 1.

Output: appends JSON-lines rows {f:[...], won:0/1, fmt:'2p'/'4p'} to --out.

Usage (on the 3060 box):
    python rl/gen_selfplay.py --games 400 --out rl/data/iter0.jsonl --workers 6
    # add the current value agent into the pool once it exists:
    python rl/gen_selfplay.py --games 400 --out rl/data/iter1.jsonl \
        --extra rl/value_agent.py --weights rl/data/value_iter0.json
"""
from __future__ import annotations
import argparse, json, os, random, sys
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "rl"))
from rl.features import extract_features  # noqa: E402

DEFAULT_POOL = [
    "single-size/main.py",
    "single-size/main_exp48.py",
    "opponents/structured.py",
    "opponents/proto1000.py",
    "opponents/orbitbotnext.py",
]
SAMPLE_FRACS = (0.2, 0.35, 0.5, 0.65, 0.8)


def _play(task):
    seed, agent_paths, weights = task
    import logging
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    if weights:
        os.environ["VALUE_WEIGHTS"] = weights
    from kaggle_environments import make
    nP = len(agent_paths)
    try:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([os.path.join(ROOT, p) if not os.path.isabs(p) else p for p in agent_paths])
    except Exception as e:
        return []
    steps = env.steps
    rew = [steps[-1][i].get("reward") for i in range(nP)]
    if any(r is None for r in rew):
        return []
    maxr = max(rew)
    won = [1 if (r == maxr and rew.count(maxr) == 1) else 0 for r in rew]
    rows = []
    N = len(steps)
    fmt = "4p" if nP >= 4 else "2p"
    for fr in SAMPLE_FRACS:
        t = min(N - 1, int(fr * (N - 1)))
        obs0 = steps[t][0]["observation"]
        for p in range(nP):
            f = extract_features(obs0, p, nP)
            if f is None:
                continue
            rows.append({"f": f, "won": won[p], "fmt": fmt})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--out", default="rl/data/iter0.jsonl")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--pool", nargs="+", default=DEFAULT_POOL)
    ap.add_argument("--extra", default=None, help="value agent path to add to pool")
    ap.add_argument("--weights", default=None, help="value weights json for --extra")
    ap.add_argument("--p4_frac", type=float, default=0.5, help="fraction of 4P games")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pool = list(args.pool)
    if args.extra:
        pool.append(args.extra)
    rng = random.Random(args.seed)
    tasks = []
    for g in range(args.games):
        nP = 4 if rng.random() < args.p4_frac else 2
        agents = [rng.choice(pool) for _ in range(nP)]
        # ensure the value-agent appears often when provided
        if args.extra and args.extra not in agents:
            agents[rng.randrange(nP)] = args.extra
        tasks.append((rng.randrange(1_000_000), agents, args.weights))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_rows = 0
    with open(args.out, "w") as fh, Pool(args.workers) as pool_:
        for rows in pool_.imap_unordered(_play, tasks, chunksize=2):
            for r in rows:
                fh.write(json.dumps(r) + "\n"); n_rows += 1
    print(f"wrote {n_rows} rows from {args.games} games -> {args.out}")


if __name__ == "__main__":
    main()
