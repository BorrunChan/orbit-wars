"""Parameter research harness: producer(tunable) vs lightver (the downloaded ipynb).

Goal: find configs that STABLY beat lightver's mechanism, and understand how each
param shifts the win-rate. 2P: producer vs lightver (both positions). 4P: 1 producer
vs 3 lightver (can the producer top a table of lightvers?). Win/rank by REAL final
score (2P=5*prod+ships, 4P=prod). Many seeds, no efficiency concern —真实优先.

Usage:
  .venv/bin/python search/vs_lightver.py --label default --seeds 12
  .venv/bin/python search/vs_lightver.py --label roi1.3 --seeds 12 --ov4 '{"roi_threshold":1.3}'
"""
from __future__ import annotations
import argparse, importlib.util, sys, os, json, dataclasses, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "search" / "gauntlet"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    sys.path.insert(0, os.path.dirname(path)); spec.loader.exec_module(m); sys.path.pop(0)
    return m


def _score(env, nP, fmt):
    obs = env.state[0]["observation"]
    w5 = 5 if fmt == "2P" else 0
    prod = [0.0] * nP; ships = [0.0] * nP
    for p in obs.get("planets", []):
        o = p[1]
        if isinstance(o, int) and 0 <= o < nP:
            ships[o] += p[5]; prod[o] += p[6]
    for f in obs.get("fleets", []):
        o = f[1] if len(f) > 1 else None
        if isinstance(o, int) and 0 <= o < nP and isinstance(f[-1], (int, float)):
            ships[o] += f[-1]
    return [w5 * prod[i] + ships[i] for i in range(nP)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="default")
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--ov4", default="{}")
    ap.add_argument("--ov2", default="{}")
    ap.add_argument("--prod", default="reyhan_1239")
    args = ap.parse_args()
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"

    prod = _load("prod_m", str(G / args.prod / "main.py"))
    lv = _load("lv_m", str(G / "lightver" / "main.py"))
    ov4 = json.loads(args.ov4); ov2 = json.loads(args.ov2)
    if ov4: prod.CONFIG_4P = dataclasses.replace(prod.CONFIG_4P, **ov4)
    if ov2: prod.CONFIG_2P = dataclasses.replace(prod.CONFIG_2P, **ov2)

    from kaggle_environments import make
    P, L = prod.agent, lv.agent

    # 2P: producer vs lightver, both positions
    w2 = l2 = d2 = 0
    for s in range(args.seeds):
        for pos in (0, 1):
            ag = [P, L] if pos == 0 else [L, P]
            env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
            sc = _score(env, 2, "2P"); ci = pos
            if sc[ci] > sc[1 - ci]: w2 += 1
            elif sc[1 - ci] > sc[ci]: l2 += 1
            else: d2 += 1

    # 4P: 1 producer + 3 lightver, producer at each rotating position
    ranks = []
    for s in range(args.seeds):
        for pos in range(4):
            ag = [L, L, L, L]; ag[pos] = P
            env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
            sc = _score(env, 4, "4P")
            me = sc[pos]
            ranks.append(1 + sum(1 for i in range(4) if i != pos and sc[i] > me))

    n2 = w2 + l2 + d2; n4 = len(ranks)
    wr2 = 100 * (w2 + 0.5 * d2) / n2
    win4 = 100 * ranks.count(1) / n4
    mr = sum(ranks) / n4
    dist = [ranks.count(k) for k in (1, 2, 3, 4)]
    print(f"{args.label:16} | 2P vs LV {wr2:5.1f}% ({w2}W-{l2}L-{d2}D) | "
          f"4P top {win4:5.1f}% mRank {mr:.2f} dist {dist} (n2={n2},n4={n4})")


if __name__ == "__main__":
    main()
