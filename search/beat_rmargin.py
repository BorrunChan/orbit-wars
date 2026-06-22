"""Measure a candidate's 2P win-rate vs rmargin (reyhan_1239), ISOLATED loading.

Objective harness for the 'beat rmargin in local 2P' task. Each agent loaded with
a unique module name + real __file__ so its OWN params.json is read (kaggle's exec
drops __file__ -> all file-path agents read cwd -> same config; see _load).

Usage: .venv/bin/python search/beat_rmargin.py <cand_dir_or_name> [seeds]
       .venv/bin/python search/beat_rmargin.py ALL [seeds]   # all gauntlet members
"""
import sys, os, importlib.util, logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
G = ROOT / "search" / "gauntlet"
REF = "reyhan_1239"


def _load(name):
    path = str(G / name / "main.py")
    mod = "ag_" + name.replace("-", "_").replace(".", "_")
    if mod in sys.modules:
        return sys.modules[mod].agent
    spec = importlib.util.spec_from_file_location(mod, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[mod] = m
    spec.loader.exec_module(m)
    return m.agent


def _score(env):
    obs = env.state[0]["observation"]
    pr = [0, 0]; sh = [0, 0]
    for p in obs["planets"]:
        o = p[1]
        if isinstance(o, int) and 0 <= o < 2:
            sh[o] += p[5]; pr[o] += p[6]
    return [5 * pr[i] + sh[i] for i in range(2)]


def vs_ref(name, seeds):
    logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
    os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
    from kaggle_environments import make
    ca = _load(name); rf = _load(REF)
    w = l = d = 0; detail = []
    for seed in range(seeds):
        for pos in (0, 1):
            ag = [ca, rf] if pos == 0 else [rf, ca]
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            env.run(ag)
            s = _score(env); ci = pos
            cs, rs = s[ci], s[1 - ci]
            if cs > rs: w += 1
            elif rs > cs: l += 1
            else: d += 1
    n = w + l + d
    return w, l, d, 100 * (w + 0.5 * d) / n


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "ALL"
    seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    names = ([p.name for p in sorted(G.iterdir()) if (p / "main.py").exists() and p.name != REF]
             if target == "ALL" else [target])
    print(f"vs {REF}, {seeds} seeds x2 pos = {seeds*2} games each (isolated):")
    rows = []
    for nm in names:
        try:
            w, l, d, wr = vs_ref(nm, seeds)
            rows.append((nm, wr, w, l, d))
        except Exception as e:
            print(f"  {nm:16} ERR {str(e)[:60]}")
    for nm, wr, w, l, d in sorted(rows, key=lambda x: -x[1]):
        flag = "  <<< BEATS rmargin" if wr > 50 else ""
        print(f"  {nm:16} {w:>2}W-{l:>2}L-{d:>2}D  win%={wr:>3.0f}%{flag}")
