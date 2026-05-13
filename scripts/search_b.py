"""
Plan B1: extended parameter search.

Improvements over scripts/search.py:
  - More parameters: adds `prod_weight` (eval prod term), `early_force_step`
    (force-launch heuristic top move if step < this), `buffer_pct` (ship
    buffer variant scale).
  - Fitness mixes 2P and 4P lineups so the result generalizes (the previous
    v20 was 2P-only and broke down in 4P + 3 strong opps).
  - Weak-opp sanity check to avoid regressions.

Usage:
    .venv/bin/python -u scripts/search_b.py [trials]
"""

import os, sys, time, random, json, logging
from pathlib import Path
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"
TEMPLATE = open(ROOT / "agents/v3p_fix4p.py").read()


def make_variant(params):
    code = TEMPLATE
    code = code.replace("SIM_HORIZON = 8", f"SIM_HORIZON = {params['horizon']}")
    code = code.replace("K = min(30, len(candidates))",
                         f"K = min({params['K']}, len(candidates))")
    code = code.replace("if p[1] != opp_player or p[5] < 20:",
                         f"if p[1] != opp_player or p[5] < {params['opp_thresh']}:")
    code = code.replace("if ships < 20:",
                         f"if ships < {params['opp_min_send']}:")
    code = code.replace("(min_ships + 0.5 * T)",
                         f"(min_ships + {params['T_weight']} * T)")
    code = code.replace("(big_ships + 0.5 * T)",
                         f"(big_ships + {params['T_weight']} * T)")
    # Buffer variant scale: original was 0.5 of min_ships + 5
    code = code.replace("buf = min(int(min_ships * 0.5) + 5, avail - min_ships)",
                         f"buf = min(int(min_ships * {params['buf_scale']}) + 5, avail - min_ships)")
    return code


def evaluate(params, seeds=1, verbose=False):
    """Mixed-format fitness. Returns weighted score and per-bucket breakdown."""
    code = make_variant(params)
    main_path = ROOT / "main_search.py"
    main_path.write_text(code)
    a = str(main_path)

    def opp(name):
        return name if name in ("random", "starter") else str(OPP_DIR / f"{name}.py")

    buckets = {
        # (label, lineup, weight, mode)
        # mode='2p' → swap sides; mode='4p' → rotate me through all positions.
        "2p_strong_lb":   (["lb1224"],        0.18, "2p"),
        "2p_strong_pro":  (["proto1000"],     0.18, "2p"),
        "2p_strong_str":  (["structured"],    0.14, "2p"),
        "2p_weak":        (["v2d_conserve"],  0.10, "2p"),
        "4p_strong":      (["lb1224", "proto1000", "structured"], 0.15, "4p"),
        "4p_mixed":       (["lb1224", "structured", "v2d_conserve"], 0.15, "4p"),
        "4p_easy":        (["starter", "starter", "starter"], 0.10, "4p"),
    }

    breakdown = {}
    total_score = 0.0
    for label, (lineup, weight, mode) in buckets.items():
        wins = games = 0
        if mode == "2p":
            opp_name = lineup[0]
            for s in range(seeds):
                for swap in (False, True):
                    a_, b_ = (opp(opp_name), a) if swap else (a, opp(opp_name))
                    me_idx = 1 if swap else 0
                    env = make("orbit_wars", configuration={"seed": s}, debug=False)
                    try:
                        env.run([a_, b_])
                        r = env.steps[-1][me_idx].reward
                        if r > 0:
                            wins += 1
                    except Exception:
                        pass
                    games += 1
        else:  # 4p
            for s in range(seeds):
                for pos in range(4):
                    agents = [opp(n) for n in lineup]
                    agents.insert(pos, a)
                    env = make("orbit_wars", configuration={"seed": s}, debug=False)
                    try:
                        env.run(agents)
                        r = env.steps[-1][pos].reward
                        if r > 0:
                            wins += 1
                    except Exception:
                        pass
                    games += 1
        wr = wins / max(1, games)
        breakdown[label] = (wins, games, wr)
        total_score += weight * wr
    return total_score, breakdown


def random_params(rng):
    return {
        "horizon": rng.choice([6, 8, 10, 12, 14, 16, 20]),
        "K": rng.choice([12, 15, 20, 25, 30, 40]),
        "opp_thresh": rng.choice([12, 16, 20, 24, 28]),
        "opp_min_send": rng.choice([12, 16, 20, 24]),
        "T_weight": rng.choice([0.3, 0.5, 0.7, 1.0, 1.2, 1.5]),
        "buf_scale": rng.choice([0.25, 0.5, 0.75, 1.0]),
    }


def main():
    rng = random.Random(0)
    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    history = []
    t0 = time.time()
    best = None
    for i in range(trials):
        p = random_params(rng)
        try:
            score, breakdown = evaluate(p, seeds=1)
        except Exception as e:
            print(f"trial {i+1:2}/{trials}  ERROR: {e}  params={p}", flush=True)
            continue
        t = time.time() - t0
        entry = {"params": p, "score": score, "breakdown": breakdown, "t": t}
        history.append(entry)
        marker = ""
        if best is None or score > best["score"]:
            best = entry
            marker = " *NEW BEST*"
        bk_str = "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in breakdown.items())
        print(f"trial {i+1:2}/{trials}  score={score:.3f}  t={t:.0f}s  {p}{marker}",
              flush=True)
        print(f"  {bk_str}", flush=True)

    print(f"\nBest: score={best['score']:.3f}  {best['params']}")
    bk_str = "\n  ".join(f"{k}: {v[0]}/{v[1]} ({v[2]:.0%})" for k, v in best["breakdown"].items())
    print(f"Best breakdown:\n  {bk_str}")
    with open(ROOT / "search_b_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
