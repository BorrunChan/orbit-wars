"""
Train behavioral cloning models per tier from bc_training.jsonl.

Models:
  1. Launch classifier: should this planet launch? (binary)
  2. Target type classifier: neutral / enemy / reinforce (given launch)
  3. Ship fraction regressor: what fraction of garrison to send (given launch)

Per tier: 800, 900, 1000+

Output: data/bc_models/{tier}_launch.pkl, {tier}_target.pkl, {tier}_ships.pkl

Usage:
    .venv/bin/python scripts/train_bc_opponents.py
"""
import json, os, pickle, sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FILE = os.path.join(ROOT, 'data', 'bc_training.jsonl')
OUT_DIR = os.path.join(ROOT, 'data', 'bc_models')
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    'step', 'n_players', 'ships', 'production', 'dist_center',
    'my_planets', 'enemy_planets', 'neutral_planets',
    'power', 'ship_ratio', 'my_prod', 'enemy_prod',
    'nearest_enemy_dist', 'nearest_enemy_ships',
    'nearest_neutral_dist', 'nearest_neutral_ships',
    'threat_ships',
]

TARGET_MAP = {'none': 0, 'neutral': 1, 'enemy': 2, 'reinforce': 3}

# Load data
print("Loading data...")
samples_by_tier = {}
for line in open(DATA_FILE):
    s = json.loads(line)
    tier = s['tier']
    if tier in ('600', '700'):
        continue  # skip weak tiers, not useful for modeling
    samples_by_tier.setdefault(tier, []).append(s)

for tier in sorted(samples_by_tier.keys()):
    print(f"\n{'='*60}")
    print(f"Training tier: {tier} ({len(samples_by_tier[tier])} samples)")
    print(f"{'='*60}")

    data = samples_by_tier[tier]

    X = np.array([[s[f] for f in FEATURE_COLS] for s in data], dtype=np.float32)
    y_launch = np.array([s['launched'] for s in data], dtype=np.int32)
    y_target = np.array([TARGET_MAP.get(s['tgt_type'], 0) for s in data], dtype=np.int32)
    y_ships = np.array([s['ship_frac'] for s in data], dtype=np.float32)

    launch_rate = y_launch.mean()
    print(f"  Launch rate: {launch_rate:.3f}")
    print(f"  Target dist (of launches): neutral={sum(y_target[y_launch==1]==1)}, "
          f"enemy={sum(y_target[y_launch==1]==2)}, reinforce={sum(y_target[y_launch==1]==3)}")
    print(f"  Avg ship_frac (of launches): {y_ships[y_launch==1].mean():.3f}")

    # Subsample for speed — max 80K samples per tier
    if len(data) > 80000:
        idx = np.random.RandomState(42).choice(len(data), 80000, replace=False)
        X = X[idx]
        y_launch = y_launch[idx]
        y_target = y_target[idx]
        y_ships = y_ships[idx]
        print(f"  Subsampled to {len(X)} samples")

    X_train, X_test, yl_train, yl_test = train_test_split(
        X, y_launch, test_size=0.15, random_state=42)

    # 1. Launch classifier
    print("  Training launch classifier...")
    launch_model = GradientBoostingClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, random_state=42)
    launch_model.fit(X_train, yl_train)
    acc = launch_model.score(X_test, yl_test)
    print(f"  Launch accuracy: {acc:.3f}")

    # 2. Target type classifier (only on launched samples)
    launched_mask = y_launch == 1
    if launched_mask.sum() > 100:
        X_launched = X[launched_mask]
        y_tgt = y_target[launched_mask]

        # Subsample if too many
        if len(X_launched) > 30000:
            idx2 = np.random.RandomState(42).choice(len(X_launched), 30000, replace=False)
            X_launched = X_launched[idx2]
            y_tgt = y_tgt[idx2]

        print("  Training target classifier...")
        target_model = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42)
        target_model.fit(X_launched, y_tgt)

        Xt_train, Xt_test, yt_train, yt_test = train_test_split(
            X_launched, y_tgt, test_size=0.15, random_state=42)
        target_model_eval = GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42)
        target_model_eval.fit(Xt_train, yt_train)
        tacc = target_model_eval.score(Xt_test, yt_test)
        print(f"  Target accuracy: {tacc:.3f}")
    else:
        target_model = None
        print("  Not enough launched samples for target model")

    # 3. Ship fraction regressor (only on launched samples)
    y_sf = y_ships[launched_mask]
    if len(y_sf) > 100:
        X_sf = X[launched_mask]
        if len(X_sf) > 30000:
            idx3 = np.random.RandomState(42).choice(len(X_sf), 30000, replace=False)
            X_sf = X_sf[idx3]
            y_sf = y_sf[idx3]

        print("  Training ship fraction regressor...")
        ships_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42)
        ships_model.fit(X_sf, y_sf)
        print(f"  Ship frac R2: {ships_model.score(X_sf, y_sf):.3f}")
    else:
        ships_model = None

    # Save
    for name, model in [('launch', launch_model), ('target', target_model), ('ships', ships_model)]:
        if model is not None:
            path = os.path.join(OUT_DIR, f'{tier}_{name}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  Saved {path}")

    # Also save feature importance for launch model
    imp = sorted(zip(FEATURE_COLS, launch_model.feature_importances_),
                 key=lambda x: -x[1])
    print(f"  Launch feature importance (top 8):")
    for feat, importance in imp[:8]:
        print(f"    {feat:<25} {importance:.3f}")

print("\nDone!")
