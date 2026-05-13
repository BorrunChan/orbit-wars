"""
Train V(state) → P(player wins) value function from round-robin game data.

Approach (inspired by lb-highest-1000-search-learned-value-function):
  - Load all games from data/round_robin_games.jsonl
  - For each (game, player, turn), build a state feature vector and label it
    with whether that player eventually won
  - Train sklearn GradientBoostingClassifier (500 trees × depth 6 by default)
  - Dump trees in pure-Python form so my agent can evaluate without sklearn

Output:
  data/value_model.py  — pure-Python module with GBC_INIT, GBC_N_TREES, GBC_TREES

Usage:
  .venv/bin/python scripts/train_value.py
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "round_robin_games.jsonl"
OUT = ROOT / "data" / "value_model.py"


def build_features(state, num_players, is_2p):
    """Extract a numeric feature vector from a state summary.

    Mirrors the lb-highest-1000 approach: ratios/leads + format indicators.
    """
    my_ships = state["my_ships"]
    opp_ships = state["opp_ships"]
    my_planets = state["my_planets"]
    opp_planets = state["opp_planets"]
    neutral_planets = state["neutral_planets"]
    my_prod = state["my_prod"]
    opp_prod = state["opp_prod"]
    my_fleet_ships = state["my_fleet_ships"]
    opp_fleet_ships = state["opp_fleet_ships"]
    my_centrality = state["my_centrality"]
    step = state["step"]

    total_planets = max(1, my_planets + opp_planets + neutral_planets)
    total_ships = max(1, my_ships + opp_ships)
    total_prod = max(1, my_prod + opp_prod)
    total_fleet_ships = max(1, my_fleet_ships + opp_fleet_ships)

    def safe_div(a, b):
        return a / b if b > 1e-9 else 0.0

    return [
        step / 500.0,
        num_players / 4.0,
        safe_div(my_planets, total_planets),
        safe_div(opp_planets, total_planets),
        safe_div(neutral_planets, total_planets),
        safe_div(my_ships, total_ships),
        safe_div(my_prod, total_prod),
        safe_div(my_fleet_ships, total_fleet_ships),
        safe_div(my_ships - opp_ships, total_ships),
        safe_div(my_planets - opp_planets, total_planets),
        safe_div(my_prod - opp_prod, total_prod),
        my_centrality / 60.0,
        1.0 if is_2p else 0.0,
        1.0 if num_players == 4 else 0.0,
        # Aggregate strength
        safe_div(my_ships + my_fleet_ships, total_ships + total_fleet_ships),
        # Per-planet density
        safe_div(my_ships, max(1, my_planets)) / 50.0,
    ]


FEATURE_NAMES = [
    "step_progress", "num_players", "my_planet_frac", "opp_planet_frac",
    "neutral_planet_frac", "my_ship_frac", "my_prod_frac", "my_fleet_frac",
    "ship_lead", "planet_lead", "prod_lead", "my_centrality",
    "is_2p", "is_4p", "my_total_strength", "my_density",
]


def load_dataset(data_path):
    """Build (X, y) for sklearn from game traces."""
    X = []
    y = []
    n_games = 0
    if not data_path.exists():
        print(f"No data at {data_path}")
        return X, y, 0

    with open(data_path) as f:
        for line in f:
            game = json.loads(line)
            if "traces" not in game:
                continue
            n_games += 1
            winner = game["winner"]
            num_players = game["num_players"]
            is_2p = num_players == 2
            for p_idx_str, traj in game["traces"].items():
                p_idx = int(p_idx_str)
                label = 1 if p_idx == winner else 0
                for turn in traj:
                    state = turn["state"]
                    feats = build_features(state, num_players, is_2p)
                    X.append(feats)
                    y.append(label)
    return X, y, n_games


def dump_trees(model, out_path):
    """Dump GBC trees to plain Python for zero-dep inference."""
    lines = [
        '"""Auto-generated GBC trees for V(state) → logit P(win).',
        'Walk: for each tree, descend until feature == -2, sum value at leaf.',
        'Add GBC_INIT initial value, then convert via sigmoid for probability."""',
        '',
    ]
    lines.append(f'GBC_INIT = {model.init_.class_prior_[1] if hasattr(model, "init_") else 0.0:.10f}')
    lines.append(f'GBC_LEARNING_RATE = {model.learning_rate}')
    lines.append(f'GBC_N_TREES = {len(model.estimators_)}')
    lines.append('')
    lines.append('# Each tree = (feature_list, threshold_list, value_list, left_list, right_list)')
    lines.append('GBC_TREES = [')

    lr = model.learning_rate
    for tree_arr in model.estimators_:
        tree = tree_arr[0].tree_
        feature = tree.feature.tolist()
        threshold = tree.threshold.tolist()
        # value at each node — multiply leaf values by learning_rate for raw
        # accumulation
        value = (tree.value.ravel() * lr).tolist()
        left = tree.children_left.tolist()
        right = tree.children_right.tolist()
        lines.append(f'    ({feature}, {[round(t, 8) for t in threshold]}, {[round(v, 10) for v in value]}, {left}, {right}),')
    lines.append(']')
    lines.append('')
    lines.append('def value_predict(feats):')
    lines.append('    """Walk all trees, return logit (raw sum). Probability = 1/(1+exp(-logit))."""')
    lines.append('    z = GBC_INIT')
    lines.append('    for feat, thr, val, left, right in GBC_TREES:')
    lines.append('        node = 0')
    lines.append('        while feat[node] >= 0:')
    lines.append('            if feats[feat[node]] <= thr[node]:')
    lines.append('                node = left[node]')
    lines.append('            else:')
    lines.append('                node = right[node]')
    lines.append('        z += val[node]')
    lines.append('    return z')

    out_path.write_text('\n'.join(lines))


def main():
    print(f"Loading {DATA}...")
    X, y, n_games = load_dataset(DATA)
    print(f"Loaded {n_games} games → {len(X)} samples")
    if not X:
        print("No data. Run round_robin.py first.")
        return

    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, brier_score_loss

    X = np.array(X)
    y = np.array(y)
    print(f"Positive rate: {y.mean():.1%}  (1 = state is from eventual winner's perspective)")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_trees = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Training GBC: {n_trees} trees × depth {depth}")

    model = GradientBoostingClassifier(
        n_estimators=n_trees,
        max_depth=depth,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
        verbose=0,
    )
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Brier: {brier:.4f}")

    # Feature importance
    importance = model.feature_importances_
    print("\nFeature importance (top 10):")
    pairs = sorted(zip(FEATURE_NAMES, importance), key=lambda kv: -kv[1])
    for name, imp in pairs[:10]:
        print(f"  {name:<22}  {imp:.3f}")

    print(f"\nDumping trees to {OUT}...")
    dump_trees(model, OUT)
    print(f"Done. File size: {OUT.stat().st_size:>9} bytes")


if __name__ == "__main__":
    main()
