"""Parameter research vs ALL downloaded ipynb opponents.

Goal: find a producer config that STABLY beats every downloaded ipynb agent.
2P: producer vs each opponent (both positions). 4P: producer + 3 opponents
(all 3-combos of the pool), producer must top the table. Win/rank by REAL final
score (2P=5*prod+ships, 4P=prod). 真实优先,不计效率.

Usage:
  .venv/bin/python search/vs_opponents.py --label default --seeds 6
  .venv/bin/python search/vs_opponents.py --label roi1.3 --seeds 6 --ov4 '{"roi_threshold":1.3}'
"""
from __future__ import annotations
import argparse, importlib.util, sys, os, json, dataclasses, logging, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "search" / "gauntlet"
OPPS = ["opp_ithe", "opp_lightver", "opp_exp48", "opp_theproducer", "opp_reyhan"]

_CACHE = {}


def _load(name):
    if name in _CACHE:
        return _CACHE[name]
    path = str(G / name / "main.py")
    spec = importlib.util.spec_from_file_location("m_" + name, path)
    m = importlib.util.module_from_spec(spec); sys.modules["m_" + name] = m
    sys.path.insert(0, str(G / name)); spec.loader.exec_module(m); sys.path.pop(0)
    _CACHE[name] = m
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
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--ov4", default="{}")
    ap.add_argument("--ov2", default="{}")
    ap.add_argument("--prod", default="reyhan_1239")
    args = ap.parse_args()
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"

    prodm = _load(args.prod)
    ov4 = json.loads(args.ov4); ov2 = json.loads(args.ov2)
    if ov4: prodm.CONFIG_4P = dataclasses.replace(prodm.CONFIG_4P, **ov4)
    if ov2: prodm.CONFIG_2P = dataclasses.replace(prodm.CONFIG_2P, **ov2)
    P = prodm.agent
    opps = {o: _load(o).agent for o in OPPS}

    from kaggle_environments import make

    # 2P: producer vs each opp
    wr2 = {}
    for oname, O in opps.items():
        w = l = d = 0
        for s in range(args.seeds):
            for pos in (0, 1):
                ag = [P, O] if pos == 0 else [O, P]
                env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
                sc = _score(env, 2, "2P"); ci = pos
                if sc[ci] > sc[1 - ci]: w += 1
                elif sc[1 - ci] > sc[ci]: l += 1
                else: d += 1
        n = w + l + d
        wr2[oname] = 100 * (w + 0.5 * d) / n

    # 4P: producer + 3 opps (all 3-combos), producer at pos0
    ranks = []
    s4 = max(2, args.seeds // 2)
    for combo in itertools.combinations(OPPS, 3):
        for s in range(s4):
            ag = [P, opps[combo[0]], opps[combo[1]], opps[combo[2]]]
            env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
            sc = _score(env, 4, "4P"); me = sc[0]
            ranks.append(1 + sum(1 for i in range(1, 4) if sc[i] > me))
    n4 = len(ranks)
    win4 = 100 * ranks.count(1) / n4
    mr = sum(ranks) / n4

    avg2 = sum(wr2.values()) / len(wr2)
    print(f"\n=== {args.label} (ov4={args.ov4} ov2={args.ov2}) ===")
    print("  2P vs each: " + " | ".join(f"{o.replace('opp_',''):10}{wr2[o]:5.1f}%" for o in OPPS))
    print(f"  2P 平均 {avg2:.1f}%  (要 >50% 才算稳赢)  | 全胜对手数: {sum(1 for v in wr2.values() if v>50)}/5")
    print(f"  4P (vs 3 opps mix): top {win4:.1f}% mRank {mr:.2f} (n={n4})")


if __name__ == "__main__":
    main()
