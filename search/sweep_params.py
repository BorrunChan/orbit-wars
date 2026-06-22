"""Single-knob parameter sweep vs all downloaded ipynb opponents.
Understand how each param shifts win-rate (esp. vs lightver, the hard one).
Producer = reyhan default; each variant overrides one knob. 真实优先.
"""
from __future__ import annotations
import importlib.util, sys, os, dataclasses, logging, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "search" / "gauntlet"
OPPS = ["opp_ithe", "opp_lightver", "opp_exp48", "opp_theproducer", "opp_reyhan"]
SEEDS = 6
_OPP_CACHE = {}
_N = [0]


def _load_fresh(name):
    _N[0] += 1
    spec = importlib.util.spec_from_file_location(f"fl_{name}_{_N[0]}", str(G / name / "main.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    sys.path.insert(0, str(G / name)); spec.loader.exec_module(m); sys.path.pop(0)
    return m


def _opp(name):
    if name not in _OPP_CACHE:
        _OPP_CACHE[name] = _load_fresh(name).agent
    return _OPP_CACHE[name]


def _score(env, nP, fmt):
    obs = env.state[0]["observation"]; w5 = 5 if fmt == "2P" else 0
    prod = [0.0] * nP; ships = [0.0] * nP
    for p in obs.get("planets", []):
        o = p[1]
        if isinstance(o, int) and 0 <= o < nP: ships[o] += p[5]; prod[o] += p[6]
    for f in obs.get("fleets", []):
        o = f[1] if len(f) > 1 else None
        if isinstance(o, int) and 0 <= o < nP and isinstance(f[-1], (int, float)): ships[o] += f[-1]
    return [w5 * prod[i] + ships[i] for i in range(nP)]


def run_variant(ov2, ov4):
    from kaggle_environments import make
    prod = _load_fresh("reyhan_1239")
    if ov4: prod.CONFIG_4P = dataclasses.replace(prod.CONFIG_4P, **ov4)
    if ov2: prod.CONFIG_2P = dataclasses.replace(prod.CONFIG_2P, **ov2)
    P = prod.agent
    wr2 = {}
    for oname in OPPS:
        O = _opp(oname); w = l = d = 0
        for s in range(SEEDS):
            for pos in (0, 1):
                ag = [P, O] if pos == 0 else [O, P]
                env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
                sc = _score(env, 2, "2P"); ci = pos
                if sc[ci] > sc[1 - ci]: w += 1
                elif sc[1 - ci] > sc[ci]: l += 1
                else: d += 1
        wr2[oname] = 100 * (w + 0.5 * d) / (w + l + d)
    ranks = []
    for combo in itertools.combinations(OPPS, 3):
        for s in range(max(2, SEEDS // 2)):
            ag = [P, _opp(combo[0]), _opp(combo[1]), _opp(combo[2])]
            env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
            sc = _score(env, 4, "4P"); me = sc[0]
            ranks.append(1 + sum(1 for i in range(1, 4) if sc[i] > me))
    return wr2, 100 * ranks.count(1) / len(ranks), sum(ranks) / len(ranks)


VARIANTS = [
    ("default", {}, {}),
    ("2p_single", {"size_multipliers": [1.0]}, {}),
    ("2p_roi1.4", {"roi_threshold": 1.4}, {}),
    ("2p_single_roi1.4", {"size_multipliers": [1.0], "roi_threshold": 1.4}, {}),
    ("roi1.3_all", {"roi_threshold": 1.3}, {"roi_threshold": 1.3}),
    ("roi1.7_all", {"roi_threshold": 1.7}, {"roi_threshold": 1.7}),
    ("beta1.5", {}, {"reinforce_size_beta": 1.5}),
    ("beta3.0", {}, {"reinforce_size_beta": 3.0}),
    ("waves9", {"max_waves_per_turn": 9}, {"max_waves_per_turn": 9}),
    ("term_roi0.8", {}, {"terminal_roi_threshold": 0.8}),
]

if __name__ == "__main__":
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    print(f"{'variant':18}{'ithe':>7}{'lightv':>7}{'exp48':>7}{'theprod':>8}{'reyhan':>7}{'2Pavg':>7}{'win#':>5}{'4Ptop':>7}")
    for label, ov2, ov4 in VARIANTS:
        wr2, win4, mr = run_variant(ov2, ov4)
        avg = sum(wr2.values()) / len(wr2); wins = sum(1 for v in wr2.values() if v > 50)
        print(f"{label:18}" + "".join(f"{wr2[o]:>7.0f}" for o in OPPS) +
              f"{avg:>7.1f}{wins:>5}{win4:>7.0f}", flush=True)
