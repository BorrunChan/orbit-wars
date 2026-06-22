"""连招最终版验证: hoard_cval(连招默认: hoard+彗星价值+三计+清仓+2P size细)
vs default producer 基线, 全量 5 对手 2P+4P, 10 seeds."""
from __future__ import annotations
import importlib.util, sys, logging, itertools
from pathlib import Path

G = Path(__file__).resolve().parent / "gauntlet"
OPPS = ["opp_ithe", "opp_lightver", "opp_exp48", "opp_theproducer", "opp_reyhan"]
SEEDS = 10
_OPP = {}; _N = [0]


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
    print(f"{'agent':18}{'ithe':>6}{'lightv':>7}{'exp48':>6}{'theP':>6}{'reyh':>6}{'2Pavg':>7}{'win#':>5}{'4Ptop':>7}", flush=True)
    for label, mod in [("default(a_comet)", "a_comet"), ("连招一体(hoard_cval)", "hoard_cval"), ("分治路由(lz_route)", "lz_route")]:
        P = _load(mod).agent; wr = {}
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
            for s in range(4):
                ag = [P, _opp(combo[0]), _opp(combo[1]), _opp(combo[2])]
                env = make("orbit_wars", configuration={"seed": s}, debug=False); env.run(ag)
                sc = _score(env, 4, "4P"); ranks.append(1 + sum(1 for i in range(1, 4) if sc[i] > sc[0]))
        avg = sum(wr.values()) / 5; wins = sum(1 for v in wr.values() if v > 50)
        print(f"{label:18}" + "".join(f"{wr[o]:>6.0f}" for o in OPPS) +
              f"{avg:>7.1f}{wins:>5}{100*ranks.count(1)/len(ranks):>7.0f}", flush=True)
