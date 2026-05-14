"""
Cluster maps by features, then analyze winner distribution per cluster.

Steps:
1. Load all games' map_features
2. Build per-game feature vector (8-10 dims)
3. KMeans cluster into K clusters
4. Per cluster: which agents win most, with what win rates
5. Output:
   - cluster centers (for runtime dispatch)
   - per-cluster best agent identification

Usage:
  .venv/bin/python scripts/cluster_maps.py [K=4]
"""

import json
import sys
import math
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "diverse_games.jsonl"
OUT = ROOT / "data" / "map_clusters.json"


def build_features(mf):
    """Map features → numeric vector. mf is map_features dict from collect."""
    av = mf.get("angular_velocity", 0)
    n_planets = mf.get("num_planets", 0)
    rotating = mf.get("rotating_planets", 0)
    avg_neut = mf.get("avg_neutral_ships", 0)
    max_neut = mf.get("max_neutral_ships", 0)
    max_prod = mf.get("max_prod", 0)
    dist_home = mf.get("dist_home_to_nearest_neutral", 0)
    prod_entropy = mf.get("prod_entropy", 0)
    prod_dist = mf.get("prod_distribution", {})
    high_prod_count = sum(v for k, v in prod_dist.items() if int(k) >= 3)
    low_prod_count = sum(v for k, v in prod_dist.items() if int(k) <= 2)
    return [
        av * 25,                              # 0.025-0.05 → 0.6-1.25
        rotating / max(1, n_planets),         # rotation ratio
        avg_neut / 50.0,                      # 0-1 normalized
        max_neut / 99.0,                      # 0-1
        max_prod / 5.0,                       # 0-1
        dist_home / 30.0,                     # 0-1+
        prod_entropy,                         # 0-2
        high_prod_count / max(1, n_planets),  # ratio of high-prod neutrals
        low_prod_count / max(1, n_planets),
        n_planets / 40.0,
    ]


FEATURE_NAMES = [
    "av_norm", "rotation_ratio", "avg_neutral_norm", "max_neutral_norm",
    "max_prod_norm", "dist_home_norm", "prod_entropy", "high_prod_ratio",
    "low_prod_ratio", "n_planets_norm",
]


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 4

    games = []
    with open(DATA) as f:
        for line in f:
            games.append(json.loads(line))
    print(f"Loaded {len(games)} games")

    # All are 2P
    games_2p = [g for g in games if g.get("num_players") == 2]
    print(f"2P games: {len(games_2p)}")
    # The new format uses 'winner_agent' instead of 'winner' (index)
    # Keep compatible with both formats below.

    # Group games by SEED (each seed = one map)
    seed_to_games = defaultdict(list)
    for g in games_2p:
        seed_to_games[g["seed"]].append(g)
    print(f"Unique seeds: {len(seed_to_games)}")

    # One feature vector per seed (use first game's map_features — same per seed)
    seeds = list(seed_to_games.keys())
    X = []
    for s in seeds:
        first_game = seed_to_games[s][0]
        X.append(build_features(first_game["map_features"]))

    import numpy as np
    from sklearn.cluster import KMeans
    X = np.array(X)
    # Z-score normalize
    X_mean = X.mean(0); X_std = X.std(0) + 1e-9
    X_norm = (X - X_mean) / X_std

    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(X_norm)
    labels = km.labels_
    centers = km.cluster_centers_  # in normalized space

    # Map seed -> cluster
    seed_to_cluster = {s: int(labels[i]) for i, s in enumerate(seeds)}

    # Per-cluster: win counts per agent
    cluster_wins = [defaultdict(lambda: [0, 0]) for _ in range(K)]  # agent -> [wins, plays]
    cluster_map_stats = [[] for _ in range(K)]
    for s, c in seed_to_cluster.items():
        for g in seed_to_games[s]:
            a, b = g["agents"]
            # support both old format (winner=idx) and new (winner_agent)
            if "winner_agent" in g:
                winner = g["winner_agent"]
            else:
                winner = g["agents"][g["winner"]]
            for ag in (a, b):
                cluster_wins[c][ag][1] += 1
            cluster_wins[c][winner][0] += 1
        cluster_map_stats[c].append(seed_to_games[s][0]["map_features"])

    print(f"\n{'='*100}")
    print(f"Cluster analysis (K={K}):")
    print(f"{'='*100}")
    for c in range(K):
        n_seeds = sum(1 for s, cc in seed_to_cluster.items() if cc == c)
        n_games = sum(plays for (_, plays) in cluster_wins[c].values())
        # Cluster center in original space
        center_orig = centers[c] * X_std + X_mean
        print(f"\n--- Cluster {c}: {n_seeds} maps, {n_games} games ---")
        print(f"  Center (denormalized):")
        for name, val in zip(FEATURE_NAMES, center_orig):
            print(f"    {name:<22} {val:.3f}")
        # Example map features summary
        if cluster_map_stats[c]:
            avs = [m["angular_velocity"] for m in cluster_map_stats[c]]
            avg_ns = [m["avg_neutral_ships"] for m in cluster_map_stats[c]]
            print(f"  Sample range:")
            print(f"    angular_velocity: {min(avs):.4f} - {max(avs):.4f}")
            print(f"    avg_neutral_ships: {min(avg_ns):.1f} - {max(avg_ns):.1f}")
        # Win rates per agent in this cluster
        print(f"  Win rates (top agents):")
        rates = []
        for ag, (w, p) in cluster_wins[c].items():
            if p == 0: continue
            rates.append((ag, w / p, w, p))
        for ag, rate, w, p in sorted(rates, key=lambda x: -x[1])[:6]:
            print(f"    {ag:<18} {w:>3}/{p:<3}  WR={rate:.0%}")

    # Save cluster info for runtime use
    out_data = {
        "K": K,
        "feature_names": FEATURE_NAMES,
        "feature_mean": X_mean.tolist(),
        "feature_std": X_std.tolist(),
        "centers_normalized": centers.tolist(),
        "centers_denormalized": (centers * X_std + X_mean).tolist(),
        "cluster_best_agent": {},
        "cluster_main_winrate": {},
    }
    # For each cluster, find best agent and main's winrate
    for c in range(K):
        rates = []
        for ag, (w, p) in cluster_wins[c].items():
            if p == 0: continue
            rates.append((ag, w / p, w, p))
        rates.sort(key=lambda x: -x[1])
        if rates:
            out_data["cluster_best_agent"][str(c)] = rates[0][0]
        main_w, main_p = cluster_wins[c].get("main", (0, 0))
        out_data["cluster_main_winrate"][str(c)] = main_w / max(1, main_p)
    out_data["seed_to_cluster"] = seed_to_cluster

    with open(OUT, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
