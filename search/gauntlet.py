"""Multi-dim h2h gauntlet: test a candidate agent vs our post-1239 submission set.

The gauntlet = every submission from reyhan/rmargin (1239, ref 53704199) ONWARD,
extracted from the exact tarballs that scored on Kaggle (search/gauntlet/<name>/).
These are strong near-peer opponents, so they discriminate better than the weak
local pool (proto1000/structured) which saturates at ~100%.

⚠️ Caveat (proven 2026-06-17): local 2P among near-peer producer configs is
ANTI-CORRELATED with Kaggle — thrift beats reyhan locally (seed0 549 vs 397)
yet scores -488 on Kaggle. And vs weak foils (proto1000) every config gives the
identical result (saturated). So this is NOT a ranker.

It IS a DISASTER SCREEN. Our real Kaggle losses are 0-planet wipes by step ~157
(precise_structured / moderate_pressure out-expand us early; see
docs/rmargin_策略说明.md loss analysis). So the Kaggle-relevant local signal is
SURVIVAL: a candidate that ends with ~1 planet / high wipe% here is a freeze /
collapse disaster (v111/thrift-style) and must NOT be submitted. A candidate
that holds ground (planets ≥ baseline, low wipe%) is merely CLEARED to submit —
Kaggle still decides the actual score.

Workflow: build candidate -> run this -> reject collapses -> submit survivors to
Kaggle (the only real judge), A/B vs reyhan 1239 = the bar.

Usage:
  .venv/bin/python search/gauntlet.py --cand search/h2h/base/main.py --seeds 5
  .venv/bin/python search/gauntlet.py --cand <dir-with-main.py> --no-4p
"""
from __future__ import annotations
import argparse, logging, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GDIR = ROOT / "search" / "gauntlet"


def members():
    return sorted(p.name for p in GDIR.iterdir() if (p / "main.py").exists())


def _score_state(env, nP, fmt):
    """Real Kaggle-style score from final observation: 2P=5*prod+ships, 4P=prod.
    env reward is binary {1,-1,0} locally and gives ZERO discrimination — this
    reads final planets/fleets and reproduces the actual scoring formula so
    config differences (hoard depth, planets held) become visible."""
    obs = env.state[0]["observation"]
    planets = obs.get("planets", [])
    fleets = obs.get("fleets", [])
    w5 = 5 if fmt == "2P" else 0
    prod = [0] * nP; ships = [0] * nP; npl = [0] * nP
    for p in planets:
        o = p[1]
        if isinstance(o, int) and 0 <= o < nP:
            npl[o] += 1; ships[o] += p[5]; prod[o] += p[6]
    for f in fleets:
        o = f[1] if len(f) > 1 else None
        if isinstance(o, int) and 0 <= o < nP:
            v = f[-1]
            if isinstance(v, (int, float)): ships[o] += v
    return [w5 * prod[i] + ships[i] for i in range(nP)], npl


def _run(agents, seed, fmt):
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    from kaggle_environments import make
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    env.run(agents)
    rew = [env.state[i]["reward"] for i in range(len(agents))]
    score, npl = _score_state(env, len(agents), fmt)
    steps = env.state[0]["observation"].get("step", 0)
    return rew, score, npl, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, help="path to candidate main.py (or its dir)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--no-4p", action="store_true")
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    cand = Path(args.cand)
    if cand.is_dir():
        cand = cand / "main.py"
    cand = str(cand.resolve())
    label = args.label or "CAND"
    opps = members()
    mpath = {m: str(GDIR / m / "main.py") for m in opps}

    print(f"candidate: {cand}")
    print(f"gauntlet: {', '.join(opps)}\n")

    # --- 2P: candidate vs each member, both positions, N seeds ---
    # NOTE: local 2P among near-peer producer configs is anti-correlated with
    # Kaggle (thrift beats reyhan locally yet is -488 on Kaggle). So W-L is NOT
    # a ranker. The Kaggle-relevant signal is SURVIVAL: our real losses are
    # 0-planet wipes (see docs), so a candidate that ends with ~1 planet here
    # (freeze/collapse, v111/thrift-style) is the disaster this screen catches.
    print("=== 2P (cand vs member) — survival screen (our losses = 0-planet wipes) ===")
    print(f"{'opponent':18} {'W-L-D':>9}  {'cand_planets':>12}  {'cand_score':>10}  {'wiped%':>6}")
    g_w = g_l = 0; all_pl = []; all_wipe = 0; all_n = 0
    for m in opps:
        w = l = d = 0; cpl = []; cs = []; wipe = 0
        for seed in range(args.seeds):
            for pos in (0, 1):
                ag = [cand, mpath[m]] if pos == 0 else [mpath[m], cand]
                rew, score, npl, steps = _run(ag, seed, "2P")
                ci, oi = pos, 1 - pos
                cpl.append(npl[ci]); cs.append(score[ci])
                if npl[ci] <= 1: wipe += 1
                if rew[ci] is not None and rew[oi] is not None:
                    if rew[ci] > rew[oi]: w += 1
                    elif rew[oi] > rew[ci]: l += 1
                    else: d += 1
        g_w += w; g_l += l; n = len(cpl)
        all_pl += cpl; all_wipe += wipe; all_n += n
        print(f"{m:18} {w:>2}-{l:>2}-{d:>2}    {sum(cpl)/n:>12.1f}  {sum(cs)/n:>10.0f}  {100*wipe/n:>5.0f}%")
    mp = sum(all_pl) / all_n if all_n else 0
    flag = "  ⚠️ COLLAPSE RISK (low planets/high wipe)" if (mp < 4 or all_wipe / max(1, all_n) > 0.3) else "  OK (holds ground)"
    print(f"\n2P: {g_w}W-{g_l}L | mean cand planets {mp:.1f} | wiped {100*all_wipe/max(1,all_n):.0f}%{flag}\n")

    if args.no_4p:
        return

    # --- 4P: candidate + 3 distinct members, rotate which 3, all seeds ---
    print("=== 4P (cand + 3 members) ===")
    import itertools
    trios = list(itertools.combinations(opps, 3))
    # cap trios to keep runtime sane; deterministic stride
    stride = max(1, len(trios) // 8)
    trios = trios[::stride][:8]
    ranks = []; wins = 0; n = 0; myplanets = []
    for trio in trios:
        for seed in range(min(args.seeds, 3)):
            ag = [cand] + [mpath[m] for m in trio]
            rew, score, npl, steps = _run(ag, seed, "4P")
            # rank by real score (prod), tiebreak by env reward
            sr = score[0]
            rank = 1 + sum(1 for x in score[1:] if x > sr)
            ranks.append(rank); n += 1; myplanets.append(npl[0])
            if rank == 1: wins += 1
    if n:
        print(f"4P games: {n}  rank1: {wins} ({100*wins/n:.0f}%)  mean_rank: {sum(ranks)/n:.2f}  "
              f"mean_myPlanets: {sum(myplanets)/n:.1f}")
    print("\nNOW SUBMIT TO KAGGLE — local is a pre-screen only (vs 1239 = the bar).")


if __name__ == "__main__":
    main()
