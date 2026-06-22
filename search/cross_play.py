"""Full round-robin cross-play among the gauntlet strategies (2P + 4P).

2P: every pair, both positions, N seeds.  4P: every 4-of-N combo, N seeds,
position rotated by seed.  Win decided by REAL final-state score
(2P=5*prod+ships, 4P=prod) because kaggle env reward is binary and useless.

Outputs a 2P standings table, a 4P standings table, and a combined ranking.
Parallel via multiprocessing.

Usage: .venv/bin/python search/cross_play.py --seeds2p 3 --seeds4p 3 --workers 8
"""
from __future__ import annotations
import argparse, itertools, os
from pathlib import Path
from multiprocessing import Pool
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
GDIR = ROOT / "search" / "gauntlet"


def members():
    return sorted(p.name for p in GDIR.iterdir() if (p / "main.py").exists())


def _score(env, nP, fmt):
    obs = env.state[0]["observation"]
    w5 = 5 if fmt == "2P" else 0
    prod = [0] * nP; ships = [0] * nP
    for p in obs.get("planets", []):
        o = p[1]
        if isinstance(o, int) and 0 <= o < nP:
            ships[o] += p[5]; prod[o] += p[6]
    for f in obs.get("fleets", []):
        o = f[1] if len(f) > 1 else None
        if isinstance(o, int) and 0 <= o < nP and isinstance(f[-1], (int, float)):
            ships[o] += f[-1]
    return [w5 * prod[i] + ships[i] for i in range(nP)]


_AGENT_CACHE = {}


def _load_isolated(name):
    """Load an agent under a UNIQUE module name so its module-level CONFIG (read
    from its own params.json via __file__) doesn't collide with another agent
    sharing the filename 'main.py'. kaggle_environments execs file agents into a
    namespace WITHOUT __file__, so agent_main falls back to os.getcwd() and every
    file-path agent ends up reading the SAME params — passing CALLABLES loaded
    here (with a real __file__) is the fix. (bug found 2026-06-17)"""
    import importlib.util, sys
    if name in _AGENT_CACHE:
        return _AGENT_CACHE[name]
    path = str(GDIR / name / "main.py")
    mod = "ag_" + name.replace("-", "_").replace(".", "_")
    spec = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    _AGENT_CACHE[name] = m.agent
    return m.agent


def _play(task):
    names, seed, fmt = task
    import logging
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    from kaggle_environments import make
    try:
        agents = [_load_isolated(n) for n in names]
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(agents)
        sc = _score(env, len(names), fmt)
    except Exception:
        return (names, None)
    return (names, sc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds2p", type=int, default=3)
    ap.add_argument("--seeds4p", type=int, default=3)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    names = members()
    print(f"strategies ({len(names)}): {', '.join(names)}\n")

    tasks = []
    # 2P: every pair, both positions, seeds
    for a, b in itertools.combinations(names, 2):
        for s in range(args.seeds2p):
            tasks.append(((a, b), s, "2P"))
            tasks.append(((b, a), s, "2P"))
    n2p = len(tasks)
    # 4P: every 4-combo, seeds, rotate positions by seed
    for combo in itertools.combinations(names, 4):
        for s in range(args.seeds4p):
            rot = tuple(combo[(i + s) % 4] for i in range(4))
            tasks.append((rot, s, "4P"))
    print(f"games: {n2p} (2P) + {len(tasks)-n2p} (4P) = {len(tasks)}; workers={args.workers}\n")

    res = []
    with Pool(args.workers) as pool:
        for r in pool.imap_unordered(_play, tasks, chunksize=2):
            res.append(r)

    # --- 2P standings ---
    rec2 = defaultdict(lambda: [0, 0, 0])  # name -> [W,L,D]
    for (ns, sc) in res:
        if sc is None or len(ns) != 2:
            continue
        a, b = ns
        if sc[0] > sc[1]: rec2[a][0] += 1; rec2[b][1] += 1
        elif sc[1] > sc[0]: rec2[b][0] += 1; rec2[a][1] += 1
        else: rec2[a][2] += 1; rec2[b][2] += 1

    print("=== 2P standings (win = higher final score 5*prod+ships) ===")
    print(f"{'strategy':18} {'W':>3} {'L':>3} {'D':>3} {'win%':>6}")
    rows2 = []
    for n in names:
        W, L, D = rec2[n]; tot = W + L + D
        wr = 100 * (W + 0.5 * D) / tot if tot else 0
        rows2.append((n, W, L, D, wr))
    for n, W, L, D, wr in sorted(rows2, key=lambda x: -x[4]):
        print(f"{n:18} {W:>3} {L:>3} {D:>3} {wr:>5.1f}%")

    # --- 4P standings ---
    rec4 = defaultdict(lambda: [0, 0.0, 0, 0])  # name -> [rank1, rank_sum, games, points]
    for (ns, sc) in res:
        if sc is None or len(ns) != 4:
            continue
        order = sorted(range(4), key=lambda i: -sc[i])
        for place, i in enumerate(order):
            nm = ns[i]
            rec4[nm][2] += 1
            rec4[nm][1] += (place + 1)
            rec4[nm][3] += (3 - place)  # 1st=3,2nd=2,3rd=1,4th=0
            if place == 0: rec4[nm][0] += 1

    print("\n=== 4P standings (rank by final prod) ===")
    print(f"{'strategy':18} {'1st':>4} {'games':>6} {'win%':>6} {'mRank':>6} {'pts':>5}")
    rows4 = []
    for n in names:
        r1, rs, g, pts = rec4[n]
        rows4.append((n, r1, g, 100 * r1 / g if g else 0, rs / g if g else 0, pts))
    for n, r1, g, wr, mr, pts in sorted(rows4, key=lambda x: -x[5]):
        print(f"{n:18} {r1:>4} {g:>6} {wr:>5.1f}% {mr:>6.2f} {pts:>5}")

    # --- combined ranking (normalized) ---
    print("\n=== combined ranking (2P win% + 4P win%, equal weight) ===")
    w2 = {n: wr for n, _, _, _, wr in rows2}
    w4 = {n: wr for n, _, _, wr, _, _ in rows4}
    comb = sorted(names, key=lambda n: -(w2[n] + w4[n]) / 2)
    print(f"{'rank':>4} {'strategy':18} {'2P win%':>8} {'4P win%':>8} {'avg':>6}")
    for i, n in enumerate(comb, 1):
        print(f"{i:>4} {n:18} {w2[n]:>7.1f}% {w4[n]:>7.1f}% {(w2[n]+w4[n])/2:>5.1f}")
    # --- 2P head-to-head matrix (row win% vs col, direct games only) ---
    h2h = defaultdict(lambda: [0, 0, 0])  # (a,b) -> [a_W, a_L, a_D]
    for (ns, sc) in res:
        if sc is None or len(ns) != 2:
            continue
        a, b = ns
        if sc[0] > sc[1]: h2h[(a, b)][0] += 1; h2h[(b, a)][1] += 1
        elif sc[1] > sc[0]: h2h[(a, b)][1] += 1; h2h[(b, a)][0] += 1
        else: h2h[(a, b)][2] += 1; h2h[(b, a)][2] += 1
    order2 = [n for n, *_ in sorted(rows2, key=lambda x: -x[4])]
    sh = {n: n[:7] for n in names}
    print("\n=== 2P head-to-head win% (row vs col) ===")
    print(f"{'':9}" + "".join(f"{sh[c]:>8}" for c in order2))
    for a in order2:
        cells = []
        for b in order2:
            if a == b: cells.append(f"{'—':>8}"); continue
            W, L, D = h2h[(a, b)]; t = W + L + D
            cells.append(f"{(100*(W+0.5*D)/t if t else 0):>7.0f}%")
        print(f"{sh[a]:9}" + "".join(cells))

    # --- 4P rank distribution + same-table pairwise dominance ---
    dist = defaultdict(lambda: [0, 0, 0, 0])  # name -> count at place 0..3
    dom = defaultdict(lambda: [0, 0])  # (a,b) -> [a_ranks_above_b, total_co_tables]
    for (ns, sc) in res:
        if sc is None or len(ns) != 4:
            continue
        place = {ns[i]: p for p, i in enumerate(sorted(range(4), key=lambda i: -sc[i]))}
        for nm, p in place.items(): dist[nm][p] += 1
        for a in ns:
            for b in ns:
                if a == b: continue
                dom[(a, b)][1] += 1
                if place[a] < place[b]: dom[(a, b)][0] += 1
    order4 = [n for n, *_ in sorted(rows4, key=lambda x: -x[5])]
    print("\n=== 4P rank distribution (1st/2nd/3rd/4th counts) ===")
    print(f"{'strategy':18} {'1st':>4} {'2nd':>4} {'3rd':>4} {'4th':>4}")
    for n in order4:
        d = dist[n]; print(f"{n:18} {d[0]:>4} {d[1]:>4} {d[2]:>4} {d[3]:>4}")
    print("\n=== 4P same-table dominance: row outranks col % (when co-tabled) ===")
    print(f"{'':9}" + "".join(f"{sh[c]:>8}" for c in order4))
    for a in order4:
        cells = []
        for b in order4:
            if a == b: cells.append(f"{'—':>8}"); continue
            w, t = dom[(a, b)]
            cells.append(f"{(100*w/t if t else 0):>7.0f}%")
        print(f"{sh[a]:9}" + "".join(cells))

    print("\n⚠️ Reminder: agents are loaded ISOLATED (unique module name) so configs "
          "genuinely differ — but win%-vs-this-pool still does NOT predict Kaggle "
          "(Pearson r=0.02 vs lightver; 5 core producers tie at 56% yet span 200 Kaggle "
          "pts). Local = crash screen ONLY. Kaggle A/B vs reyhan 1239 is the judge.")


if __name__ == "__main__":
    main()
