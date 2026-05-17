"""
Train V(state) → P(I win) classifier for 4P games (and 2P if data has it).

Designed for GPU XGBoost (CUDA-accelerated tree training).

Features (16 dims):
  - step_progress (step / 500)
  - is_2p, is_4p (one-hot game format)
  - my_ship_frac, my_planet_frac, my_prod_frac
  - my_centrality_norm
  - max_opp_ship_frac, max_opp_planet_frac, max_opp_prod_frac
  - opp_count_alive (how many opps still have planets)
  - min_enemy_dist_norm
  - ship_lead, planet_lead, prod_lead (normalized)
  - my_fleet_frac (ships in flight / total)

Label: 1 if (state's player) is the eventual game winner, else 0.

Output: data/value_4p_model.py — pure-Python tree dump for inference.

Usage:
  pip install xgboost numpy scikit-learn
  python scripts/train_value_4p.py [n_trees=500] [depth=6] [--gpu]
"""

import argparse, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_4P = ROOT / "data" / "v4p_games.jsonl"
DATA_2P_OPT = ROOT / "data" / "shot_games.jsonl"  # optionally mix in 2P data
OUT = ROOT / "data" / "value_4p_model.py"


FEATURE_NAMES = [
    "step_progress", "is_2p", "is_4p",
    "my_ship_frac", "my_planet_frac", "my_prod_frac", "my_centrality",
    "max_opp_ship_frac", "max_opp_planet_frac", "max_opp_prod_frac",
    "opp_count_alive", "min_enemy_dist_norm",
    "ship_lead", "planet_lead", "prod_lead", "my_fleet_frac",
]


def build_features_4p(state_dict, num_players):
    """From compact_state output (4P collector format)."""
    step = state_dict["step"]
    my = state_dict["my"]
    opps = state_dict["opps"]

    my_ships = my["ships"]
    my_planets = my["planets"]
    my_prod = my["prod"]
    my_fleet_ships = my["fleet_ships"]
    my_centrality = my["centrality"]
    min_enemy_dist = state_dict.get("min_enemy_dist", 100.0)

    opp_ships = sum(o["ships"] for o in opps)
    opp_planets = sum(o["planets"] for o in opps)
    opp_prod = sum(o["prod"] for o in opps)
    opp_fleet_ships = sum(o["fleet_ships"] for o in opps)
    max_opp_ships = max((o["ships"] for o in opps), default=0)
    max_opp_planets = max((o["planets"] for o in opps), default=0)
    max_opp_prod = max((o["prod"] for o in opps), default=0)
    opp_count_alive = sum(1 for o in opps if o["planets"] > 0 or o["ships"] > 0)

    total_ships = max(1, my_ships + opp_ships)
    total_planets = max(1, my_planets + opp_planets + state_dict.get("neutral_planets", 0))
    total_prod = max(1, my_prod + opp_prod)
    total_fleet_ships = max(1, my_fleet_ships + opp_fleet_ships)

    def sd(a, b):
        return a / b if b > 1e-9 else 0.0

    return [
        step / 500.0,
        1.0 if num_players == 2 else 0.0,
        1.0 if num_players == 4 else 0.0,
        sd(my_ships, total_ships),
        sd(my_planets, total_planets),
        sd(my_prod, total_prod),
        my_centrality / 60.0,
        sd(max_opp_ships, total_ships),
        sd(max_opp_planets, total_planets),
        sd(max_opp_prod, total_prod),
        opp_count_alive / 3.0,
        min(1.0, min_enemy_dist / 100.0),
        sd(my_ships - opp_ships, total_ships),
        sd(my_planets - opp_planets, total_planets),
        sd(my_prod - opp_prod, total_prod),
        sd(my_fleet_ships, total_fleet_ships),
    ]


def load_4p_data():
    X, y = [], []
    if not DATA_4P.exists():
        return X, y, 0
    n_games = 0
    with open(DATA_4P) as f:
        for line in f:
            g = json.loads(line)
            n_games += 1
            winner = g["winner"]
            num_players = g.get("num_players", 4)
            for p_idx_str, traj in g["traces"].items():
                p_idx = int(p_idx_str)
                label = 1 if p_idx == winner else 0
                for turn in traj:
                    state = turn["state"]
                    feats = build_features_4p(state, num_players)
                    X.append(feats)
                    y.append(label)
    return X, y, n_games


def dump_trees_xgboost(booster, out_path, feature_names):
    """Dump XGBoost trees to pure-Python form (zero-dep inference)."""
    trees_data = booster.get_dump(dump_format='json')
    lines = [
        '"""Auto-generated V(state) classifier (4P + 2P).',
        'Walk: sum tree leaf values + bias. Sigmoid for probability."""',
        '',
    ]
    # Base score / bias
    base_score = float(booster.attr("base_score") or 0.5)
    lines.append(f'V_BASE = {math.log(base_score / (1 - base_score)) if 0 < base_score < 1 else 0.0:.10f}')
    lines.append(f'V_N_TREES = {len(trees_data)}')
    lines.append('')
    lines.append('# Each tree: (feature_idx_list, threshold_list, value_list, left_list, right_list)')
    lines.append('V_TREES = [')
    for tree_json in trees_data:
        tree = json.loads(tree_json)
        nodes = []
        # BFS flatten
        def flatten(node, idx_map):
            if 'leaf' in node:
                nodes.append({"feat": -2, "thr": 0, "val": node["leaf"],
                               "left": -1, "right": -1})
                return len(nodes) - 1
            cur_idx = len(nodes)
            nodes.append(None)  # placeholder
            feat_str = node["split"]
            # XGBoost uses "f<idx>" or named features
            if feat_str.startswith("f"):
                feat = int(feat_str[1:])
            else:
                feat = feature_names.index(feat_str)
            thr = node["split_condition"]
            # Children: "yes" branch is < threshold, "no" is >= threshold
            yes_id = node["yes"]
            no_id = node["no"]
            children_by_id = {c["nodeid"]: c for c in node["children"]}
            left_idx = flatten(children_by_id[yes_id], idx_map)
            right_idx = flatten(children_by_id[no_id], idx_map)
            nodes[cur_idx] = {"feat": feat, "thr": thr, "val": 0,
                               "left": left_idx, "right": right_idx}
            return cur_idx
        flatten(tree, {})
        feat_list = [n["feat"] for n in nodes]
        thr_list = [round(n["thr"], 6) for n in nodes]
        val_list = [round(n["val"], 8) for n in nodes]
        left_list = [n["left"] for n in nodes]
        right_list = [n["right"] for n in nodes]
        lines.append(f'    ({feat_list}, {thr_list}, {val_list}, {left_list}, {right_list}),')
    lines.append(']')
    lines.append('')
    lines.append('import math as _math')
    lines.append('def value_predict(feats):')
    lines.append('    """Return P(I win). Walk all trees, sum, sigmoid."""')
    lines.append('    z = V_BASE')
    lines.append('    for feat, thr, val, left, right in V_TREES:')
    lines.append('        node = 0')
    lines.append('        while feat[node] >= 0:')
    lines.append('            if feats[feat[node]] < thr[node]:')
    lines.append('                node = left[node]')
    lines.append('            else:')
    lines.append('                node = right[node]')
    lines.append('        z += val[node]')
    lines.append('    return 1.0 / (1.0 + _math.exp(-z))')
    out_path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=500)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--gpu", action="store_true",
                    help="Use GPU acceleration (requires CUDA)")
    args = ap.parse_args()

    print(f"Loading {DATA_4P}...")
    X, y, n_games = load_4p_data()
    print(f"Loaded {n_games} 4P games → {len(X)} samples")

    if not X:
        print("No data. Run collect_4p.py first.")
        return

    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        return
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, brier_score_loss

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Positive rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": args.depth,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
    }
    if args.gpu:
        # xgboost 2.0+ syntax
        params["device"] = "cuda"

    print(f"Training XGBoost: {args.trees} trees × depth {args.depth}, "
          f"GPU={args.gpu}")
    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=FEATURE_NAMES)
    booster = xgb.train(
        params, dtrain,
        num_boost_round=args.trees,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=50,
    )
    p_te = booster.predict(dtest)
    auc = roc_auc_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)
    print(f"\nValidation AUC: {auc:.4f}")
    print(f"Validation Brier: {brier:.4f}")

    # Feature importance
    importance = booster.get_score(importance_type='gain')
    print("\nFeature importance (top 10):")
    for name, score in sorted(importance.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:<22}  {score:.1f}")

    print(f"\nDumping trees to {OUT}...")
    dump_trees_xgboost(booster, OUT, FEATURE_NAMES)
    print(f"File size: {OUT.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
