"""
Random search over agent hyperparameters. Each config evaluated by 2 seeds
vs 3 strong opponents (12 games). Best config wins.

Generates parameterized main_search.py and benches each.
"""

import os, sys, time, random, json, logging
from pathlib import Path
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"

TEMPLATE = open(ROOT / "agents/v3p_fix4p.py").read()

def make_variant(params):
    """Inject params into template."""
    code = TEMPLATE
    # SIM_HORIZON
    code = code.replace("SIM_HORIZON = 8", f"SIM_HORIZON = {params['horizon']}")
    # K
    code = code.replace("K = min(30, len(candidates))",
                         f"K = min({params['K']}, len(candidates))")
    # opp_starter threshold (in p[5] < 20 line)
    code = code.replace("if p[1] != opp_player or p[5] < 20:",
                         f"if p[1] != opp_player or p[5] < {params['opp_thresh']}:")
    code = code.replace("if ships < 20:\n            continue",
                         f"if ships < {params['opp_min_send']}:\n            continue")
    # Score formula T weight
    code = code.replace("(min_ships + 0.5 * T)",
                         f"(min_ships + {params['T_weight']} * T)")
    code = code.replace("(big_ships + 0.5 * T)",
                         f"(big_ships + {params['T_weight']} * T)")
    return code

def evaluate(params, seeds=2):
    code = make_variant(params)
    main_path = ROOT / "main_search.py"
    main_path.write_text(code)
    # Quick 2-seed bench vs 3 opps
    wins = 0; games = 0
    for opp_name in ["lb1224", "proto1000", "structured"]:
        opp_path = str(OPP_DIR / f"{opp_name}.py")
        for s in range(seeds):
            for swap in (False, True):
                a = (opp_path, str(main_path)) if swap else (str(main_path), opp_path)
                me_idx = 1 if swap else 0
                env = make("orbit_wars", configuration={"seed": s}, debug=False)
                try:
                    env.run(list(a))
                    r = env.steps[-1][me_idx].reward
                    if r > 0:
                        wins += 1
                    games += 1
                except Exception:
                    games += 1
    return wins, games

def random_params(rng):
    # Focused: stay near the winning region from initial search.
    return {
        "horizon": rng.choice([12, 14, 16, 20, 24]),
        "K": rng.choice([12, 15, 18, 22]),
        "opp_thresh": rng.choice([12, 16, 20, 24, 28]),
        "opp_min_send": rng.choice([16, 20, 24]),
        "T_weight": rng.choice([0.7, 1.0, 1.2, 1.5, 2.0]),
    }

def main():
    rng = random.Random(0)
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    history = []
    t0 = time.time()
    best = None
    for i in range(trials):
        p = random_params(rng)
        wins, games = evaluate(p)
        wr = wins / max(1, games)
        t = time.time() - t0
        history.append({"params": p, "wins": wins, "games": games, "wr": wr, "t": t})
        marker = ""
        if best is None or wr > best["wr"]:
            best = history[-1]; marker = " *NEW BEST*"
        print(f"trial {i+1:2}/{trials}  wins={wins}/{games}  WR={wr:.0%}  t={t:.0f}s  {p}{marker}")
    print(f"\nBest: {best['params']} -> WR={best['wr']:.0%}")
    with open(ROOT / "search_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
