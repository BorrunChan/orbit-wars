"""全面单旋钮扫参 vs 所有 ipynb 对手. 从干净 default base 出发(脚本强制,不依赖
params), 5 类每个旋钮单独改一个, 看对 vs 各对手胜率的影响. 真实优先.
"""
from __future__ import annotations
import importlib.util, sys, dataclasses, logging, itertools
from pathlib import Path

G = Path(__file__).resolve().parent / "gauntlet"
OPPS = ["opp_ithe", "opp_lightver", "opp_exp48", "opp_theproducer", "opp_reyhan"]
SEEDS = 6
_OPP = {}; _N = [0]

# 干净 default base (4P) — 强制覆盖, 不管 a_comet 读到的 params
DEFAULT = {
    "roi_threshold": 1.5, "reinforce_size_beta": 2.2, "size_multipliers": (0.5, 0.75, 1.0),
    "horizon": 13, "max_offensive_targets": 12, "max_sources_per_lane": 6,
    "min_ships_to_launch": 4.0, "max_waves_per_turn": 7, "defense_reserve_beta": 0.0,
    "enable_comet": False, "enable_opportunist": False, "enable_chenghuo": False,
    "enable_evacuation": False, "evac_counterstrike": False, "enable_defense": False,
    "enable_prod_rush": False, "geometry_weight": 0.0, "hoard_min_planets": 0, "hoard_roi_mult": 1.0,
}

_C = {"enable_comet": True}   # 彗星跳板恒开, 研究它配什么参数最强
VARIANTS = [
    ("no_comet对照", {}),                         # 不开彗星 (看彗星纯增量)
    ("comet_only基准", {**_C}),                   # 仅彗星跳板 = 对照基准
    # 彗星 + ① 兵法
    ("comet+欲擒", {**_C, "enable_opportunist": True}),
    ("comet+趁火", {**_C, "enable_chenghuo": True}),
    ("comet+出逃", {**_C, "enable_evacuation": True}),
    ("comet+围魏", {**_C, "enable_evacuation": True, "evac_counterstrike": True}),
    # 彗星 + ② hoard
    ("comet+hoard15", {**_C, "hoard_min_planets": 15, "hoard_roi_mult": 2.0}),
    # 彗星 + ③ i-the-orbit
    ("comet+防御", {**_C, "enable_defense": True}),
    ("comet+经济冲刺", {**_C, "enable_prod_rush": True}),
    ("comet+中心性", {**_C, "geometry_weight": 0.35}),
    # 彗星 + ④ 保守4P
    ("comet+roi1.3", {**_C, "roi_threshold": 1.3}),
    ("comet+roi1.7", {**_C, "roi_threshold": 1.7}),
    ("comet+beta1.5", {**_C, "reinforce_size_beta": 1.5}),
    ("comet+beta3.0", {**_C, "reinforce_size_beta": 3.0}),
    # 彗星 + ⑤ 结构
    ("comet+horizon17", {**_C, "horizon": 17}),
    ("comet+waves9", {**_C, "max_waves_per_turn": 9}),
    ("comet+maxoff14", {**_C, "max_offensive_targets": 14}),
    ("comet+maxsrc12", {**_C, "max_sources_per_lane": 12}),
    # 彗星抢占强度
    ("comet_bonus8(更狠抢)", {**_C, "comet_prod_bonus": 8.0}),
    ("comet_bonus2(轻抢)", {**_C, "comet_prod_bonus": 2.0}),
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
    print(f"{'variant':16}{'ithe':>6}{'lightv':>7}{'exp48':>6}{'theP':>6}{'reyh':>6}{'2Pavg':>7}{'win#':>5}{'4Ptop':>7}", flush=True)
    for label, ov in VARIANTS:
        pm = _load("a_comet")   # params.json removed -> agent_main default base (2P/4P correct)
        if ov:
            pm.CONFIG_4P = dataclasses.replace(pm.CONFIG_4P, **ov)
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
        ranks = []
        for combo in itertools.combinations(OPPS, 3):
            for s in range(3):
                ag = [P, _opp(combo[0]), _opp(combo[1]), _opp(combo[2])]
                env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
                sc = _score(env, 4, "4P"); ranks.append(1 + sum(1 for i in range(1, 4) if sc[i] > sc[0]))
        avg = sum(wr.values()) / 5; wins = sum(1 for v in wr.values() if v > 50)
        print(f"{label:16}" + "".join(f"{wr[o]:>6.0f}" for o in OPPS) +
              f"{avg:>7.1f}{wins:>5}{100*ranks.count(1)/len(ranks):>7.0f}", flush=True)
