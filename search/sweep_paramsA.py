"""支线A: 纯参数扫参 (无彗星) vs 所有 ipynb 对手. base = agent_main default.
PART1: 4P 旋钮 (override CONFIG_4P) -> 4P 混战夺冠率.
PART2: 2P 旋钮 (override CONFIG_2P, 用 2P 正确 default) -> 2P 单挑各对手胜率.
"""
from __future__ import annotations
import importlib.util, sys, dataclasses, logging, itertools
from pathlib import Path

G = Path(__file__).resolve().parent / "gauntlet"
OPPS = ["opp_ithe", "opp_lightver", "opp_exp48", "opp_theproducer", "opp_reyhan"]
SEEDS = 6
_OPP = {}; _N = [0]

V4 = [
    ("default", {}),
    ("①出逃", {"enable_evacuation": True}),
    ("①出逃+围魏", {"enable_evacuation": True, "evac_counterstrike": True}),
    ("①欲擒", {"enable_opportunist": True}),
    ("①趁火", {"enable_chenghuo": True}),
    ("②hoard15", {"hoard_min_planets": 15, "hoard_roi_mult": 2.0}),
    ("③防御", {"enable_defense": True}),
    ("③经济冲刺", {"enable_prod_rush": True}),
    ("③中心性", {"geometry_weight": 0.35}),
    ("④roi1.3", {"roi_threshold": 1.3}),
    ("④roi1.7", {"roi_threshold": 1.7}),
    ("④reserve0.4", {"defense_reserve_beta": 0.4}),
    ("④beta1.5", {"reinforce_size_beta": 1.5}),
    ("④beta3.0", {"reinforce_size_beta": 3.0}),
    ("⑤size单1.0", {"size_multipliers": (1.0,)}),
    ("⑤horizon11", {"horizon": 11}),
    ("⑤horizon17", {"horizon": 17}),
    ("⑤maxoff14", {"max_offensive_targets": 14}),
    ("⑤maxsrc12", {"max_sources_per_lane": 12}),
    ("⑤min3", {"min_ships_to_launch": 3.0}),
    ("⑤waves9", {"max_waves_per_turn": 9}),
]
# 2P: 没有兵法/彗星 (1v1), 主要结构+roi+beta. base 2P default: horizon18/roi1.5/size(.5,.75,1)/maxsrc12
V2 = [
    ("default", {}),
    ("roi1.3", {"roi_threshold": 1.3}),
    ("roi1.4", {"roi_threshold": 1.4}),
    ("roi1.7", {"roi_threshold": 1.7}),
    ("size单1.0", {"size_multipliers": (1.0,)}),
    ("size.33.66.1", {"size_multipliers": (0.33, 0.66, 1.0)}),
    ("horizon14", {"horizon": 14}),
    ("horizon22", {"horizon": 22}),
    ("beta1.5", {"reinforce_size_beta": 1.5}),
    ("beta3.0", {"reinforce_size_beta": 3.0}),
    ("waves9", {"max_waves_per_turn": 9}),
    ("min3", {"min_ships_to_launch": 3.0}),
    ("maxsrc8", {"max_sources_per_lane": 8}),
]


def _load(name):
    _N[0] += 1
    spec = importlib.util.spec_from_file_location(f"x{name}{_N[0]}", str(G / name / "main.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    sys.path.insert(0, str(G / name)); spec.loader.exec_module(m); sys.path.pop(0)
    return m


def _opp(n):
    if n not in _OPP: _OPP[n] = _load(n).agent
    return _OPP[n]


def _score(env, nP, fmt):
    obs = env.state[0]["observation"]; w5 = 5 if fmt == "2P" else 0
    pr = [0.0] * nP; sh = [0.0] * nP
    for p in obs.get("planets", []):
        o = p[1]
        if isinstance(o, int) and 0 <= o < nP: sh[o] += p[5]; pr[o] += p[6]
    for f in obs.get("fleets", []):
        o = f[1] if len(f) > 1 else None
        if isinstance(o, int) and 0 <= o < nP and isinstance(f[-1], (int, float)): sh[o] += f[-1]
    return [w5 * pr[i] + sh[i] for i in range(nP)]


if __name__ == "__main__":
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    from kaggle_environments import make

    print("=== PART1: 4P 旋钮 -> 4P 混战夺冠率 (1producer+3opp) ===", flush=True)
    print(f"{'variant':16}{'4Ptop':>7}", flush=True)
    for label, ov in V4:
        pm = _load("a_comet")
        if ov: pm.CONFIG_4P = dataclasses.replace(pm.CONFIG_4P, **ov)
        P = pm.agent; ranks = []
        for combo in itertools.combinations(OPPS, 3):
            for s in range(3):
                ag = [P, _opp(combo[0]), _opp(combo[1]), _opp(combo[2])]
                env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
                sc = _score(env, 4, "4P"); ranks.append(1 + sum(1 for i in range(1, 4) if sc[i] > sc[0]))
        print(f"{label:16}{100*ranks.count(1)/len(ranks):>7.0f}", flush=True)

    print("\n=== PART2: 2P 旋钮 -> 2P 单挑各对手胜率 ===", flush=True)
    print(f"{'variant':14}{'ithe':>6}{'lightv':>7}{'exp48':>6}{'theP':>6}{'reyh':>6}{'avg':>6}{'win#':>5}", flush=True)
    for label, ov in V2:
        pm = _load("a_comet")
        if ov: pm.CONFIG_2P = dataclasses.replace(pm.CONFIG_2P, **ov)
        P = pm.agent; wr = {}
        for on in OPPS:
            O = _opp(on); w = l = d = 0
            for s in range(SEEDS):
                for pos in (0, 1):
                    ag = [P, O] if pos == 0 else [O, P]
                    env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
                    sc = _score(env, 2, "2P"); ci = pos
                    if sc[ci] > sc[1 - ci]: w += 1
                    elif sc[1 - ci] > sc[ci]: l += 1
                    else: d += 1
            wr[on] = 100 * (w + 0.5 * d) / (w + l + d)
        avg = sum(wr.values()) / 5; wins = sum(1 for v in wr.values() if v > 50)
        print(f"{label:14}" + "".join(f"{wr[o]:>6.0f}" for o in OPPS) + f"{avg:>6.1f}{wins:>5}", flush=True)
