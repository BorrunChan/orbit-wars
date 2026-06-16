"""Config-search factory for the Producer engine (à la reyhan-ksatria).

Generates param candidates along axes from a base config, pre-screens each
LOCALLY by win-rate vs a fixed opponent pool (multiprocess), ranks them. The
local signal is a PRE-SCREEN only (Kaggle bench is anti-correlated + ~15-episode
noisy) — its job is to prune disasters and surface a short list; final ranking
is Kaggle. Build + submit the top few with build_submission.py.

Engine = search/agent_main.py (reyhan's parametrized exp48 + reinforce_size_beta,
reads params.json next to it). Each candidate = a dir with main.py+orbit_lite+params.

Usage:
    python search/search.py --games 16 --workers 6 --out search/results.json
    python search/search.py --only-generate          # just list candidates
"""
from __future__ import annotations
import argparse, json, os, shutil, sys, random, copy
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH = os.path.join(ROOT, "search")
CANDDIR = os.path.join(SEARCH, "_cand")
ORBIT = os.path.join(ROOT, "single-size", "orbit_lite")
# HARD pool only — weak opponents (structured) saturate the screen at ~100% and
# give no discrimination. proto1000 is the 2P structural nemesis.
POOL_4P = ["opponents/proto1000.py", "opponents/orbitbotnext.py", "opponents/mlhybrid.py", "opponents/lb1224.py"]
POOL_2P = ["opponents/proto1000.py"]

# search axes: (config_key, param, [values]). Applied one-at-a-time onto base.
AXES = [
    ("both", "reinforce_size_beta", [1.0, 2.2, 4.0]),
    ("both", "min_ships_to_launch", [3.0, 6.0]),
    ("both", "max_regroup_targets_per_source", [4]),
    ("both", "max_regroup_time", [5.0]),
    ("4p", "horizon", [11, 15]),
    ("4p", "max_offensive_targets", [12, 14]),
    ("4p", "max_sources_per_lane", [8]),
    ("4p", "size_multipliers", [[0.33, 0.66, 1.0], [1.0]]),
    ("2p", "size_multipliers", [[1.0]]),
    ("both", "roi_threshold", [1.3, 1.7]),
]


def gen_candidates(base):
    cands = {"base": copy.deepcopy(base)}
    for scope, key, vals in AXES:
        for v in vals:
            c = copy.deepcopy(base)
            for cfg in (["config_2p", "config_4p"] if scope == "both" else
                        ["config_2p"] if scope == "2p" else ["config_4p"]):
                c[cfg][key] = v
            tag = f"{key}={v if not isinstance(v, list) else 'x'.join(str(x) for x in v)}_{scope}"
            cands[tag] = c
    # a couple of reyhan-flavored combos
    combo = copy.deepcopy(base)
    combo["config_2p"].update(min_ships_to_launch=3.0, max_regroup_targets_per_source=4, max_regroup_time=5.0, reinforce_size_beta=2.2)
    combo["config_4p"].update(min_ships_to_launch=3.0, max_regroup_targets_per_source=4, max_regroup_time=5.0,
                              horizon=15, max_offensive_targets=14, max_sources_per_lane=8, size_multipliers=[0.33, 0.66, 1.0])
    cands["reyhan_like"] = combo
    return cands


def setup_cand_dirs(cands):
    if os.path.exists(CANDDIR):
        shutil.rmtree(CANDDIR)
    os.makedirs(CANDDIR)
    paths = {}
    for tag, params in cands.items():
        d = os.path.join(CANDDIR, tag.replace("/", "_"))
        os.makedirs(d)
        shutil.copy2(os.path.join(SEARCH, "agent_main.py"), os.path.join(d, "main.py"))
        json.dump(params, open(os.path.join(d, "params.json"), "w"))
        os.symlink(ORBIT, os.path.join(d, "orbit_lite"))
        paths[tag] = os.path.join(d, "main.py")
    return paths


def _play(task):
    tag, main_path, seed, pos, opps = task
    import logging
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    from kaggle_environments import make
    nP = len(opps) + 1
    agents = [os.path.join(ROOT, o) for o in opps]
    agents.insert(pos, main_path)
    try:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
    except Exception:
        return (tag, None)
    rew = [env.state[i]["reward"] for i in range(nP)]
    if rew[pos] is None:
        return (tag, None)
    mx = max(r for r in rew if r is not None)
    return (tag, 1 if (rew[pos] == mx and rew.count(mx) == 1) else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=16, help="games per candidate")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="search/results.json")
    ap.add_argument("--p4_frac", type=float, default=0.5)
    ap.add_argument("--only-generate", action="store_true")
    args = ap.parse_args()

    base = json.load(open(os.path.join(SEARCH, "base_params.json")))
    cands = gen_candidates(base)
    print(f"candidates: {len(cands)}")
    for t in cands:
        print("  ", t)
    if args.only_generate:
        return
    paths = setup_cand_dirs(cands)

    rng = random.Random(0)
    tasks = []
    for tag, mp in paths.items():
        for g in range(args.games):
            nP = 4 if rng.random() < args.p4_frac else 2
            opps = rng.sample(POOL_4P, 3) if nP == 4 else list(POOL_2P)
            pos = rng.randrange(nP)
            tasks.append((tag, mp, rng.randrange(1_000_000), pos, opps))
    print(f"running {len(tasks)} games on {args.workers} workers...")

    agg = {t: [0, 0] for t in paths}  # wins, n
    with Pool(args.workers) as pool:
        for tag, r in pool.imap_unordered(_play, tasks, chunksize=4):
            if r is None:
                continue
            agg[tag][0] += r; agg[tag][1] += 1

    rows = sorted(((t, w, n, (w / n if n else 0)) for t, (w, n) in agg.items()),
                  key=lambda x: -x[3])
    json.dump({t: {"wins": w, "n": n, "wr": wr} for t, w, n, wr in rows}, open(args.out, "w"), indent=2)
    print("\n=== ranked (local pre-screen win-rate vs pool) ===")
    for t, w, n, wr in rows:
        print(f"  {wr*100:5.1f}%  ({w}/{n})  {t}")
    print(f"\nwrote {args.out}. Build+submit the top few; Kaggle is the real judge.")


if __name__ == "__main__":
    main()
