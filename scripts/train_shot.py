"""
Train shot-success classifier from data/shot_outcomes_v2.jsonl.

Output: data/shot_model.py — pure-Python tree dump for inference.

Usage:
  .venv/bin/python scripts/train_shot.py [n_trees=200] [depth=4]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "shot_outcomes_v2.jsonl"
OUT = ROOT / "data" / "shot_model.py"


FEATURE_NAMES = [
    "src_ships", "src_prod", "tgt_ships", "tgt_prod", "tgt_owner_type",
    "distance", "fleet_size", "fleet_speed", "eta", "predicted_garrison",
    "margin", "my_planets_count", "my_total_ships", "opp_total_ships",
    "step_progress", "num_players_norm", "is_2p",
]


def load():
    X, y = [], []
    with open(DATA) as f:
        for line in f:
            d = json.loads(line)
            X.append(d["feats"])
            y.append(d["label"])
    return X, y


def dump_trees(model, out_path):
    lines = [
        '"""Auto-generated shot-success classifier.',
        'Walk: for each tree, descend until feature == -2, sum value at leaf.',
        'Sum + GBC_INIT = raw logit. Sigmoid for P(capture success)."""',
        '',
    ]
    init = float(model.init_.class_prior_[1]) if hasattr(model, "init_") else 0.0
    lines.append(f'SHOT_INIT = {init:.10f}')
    lines.append(f'SHOT_N_TREES = {len(model.estimators_)}')
    lines.append('')
    lines.append('SHOT_TREES = [')
    lr = model.learning_rate
    for tree_arr in model.estimators_:
        tree = tree_arr[0].tree_
        feat = tree.feature.tolist()
        thr = tree.threshold.tolist()
        val = (tree.value.ravel() * lr).tolist()
        left = tree.children_left.tolist()
        right = tree.children_right.tolist()
        lines.append(f'    ({feat}, {[round(t, 8) for t in thr]}, '
                     f'{[round(v, 10) for v in val]}, {left}, {right}),')
    lines.append(']')
    lines.append('')
    lines.append('def shot_predict(feats):')
    lines.append('    """Return raw logit. Probability = 1/(1+exp(-logit))."""')
    lines.append('    z = SHOT_INIT')
    lines.append('    for feat, thr, val, left, right in SHOT_TREES:')
    lines.append('        node = 0')
    lines.append('        while feat[node] >= 0:')
    lines.append('            if feats[feat[node]] <= thr[node]:')
    lines.append('                node = left[node]')
    lines.append('            else:')
    lines.append('                node = right[node]')
    lines.append('        z += val[node]')
    lines.append('    return z')
    out_path.write_text("\n".join(lines))


def main():
    if not DATA.exists():
        print(f"No data at {DATA}")
        return
    import numpy as np
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, brier_score_loss

    X, y = load()
    X = np.array(X)
    y = np.array(y)
    print(f"Loaded {len(X)} shots")
    print(f"Positive (success) rate: {y.mean():.1%}")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    n_trees = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    print(f"Training GBC: {n_trees}T × d{depth}")

    model = GradientBoostingClassifier(
        n_estimators=n_trees, max_depth=depth, learning_rate=0.05,
        subsample=0.8, random_state=42, verbose=0,
    )
    model.fit(X_tr, y_tr)

    p_te = model.predict_proba(X_te)[:, 1]
    auc = roc_auc_score(y_te, p_te)
    brier = brier_score_loss(y_te, p_te)
    print(f"Validation AUC: {auc:.4f}")
    print(f"Validation Brier: {brier:.4f}")

    importance = model.feature_importances_
    print("\nFeature importance (top 10):")
    pairs = sorted(zip(FEATURE_NAMES, importance), key=lambda kv: -kv[1])
    for name, imp in pairs[:10]:
        print(f"  {name:<22}  {imp:.3f}")

    dump_trees(model, OUT)
    print(f"\nWrote {OUT}  ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
