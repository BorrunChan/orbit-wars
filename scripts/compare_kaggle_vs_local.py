"""
Compare distributions:
  - Kaggle real opponents (typeIIIfairy, bowwowforeach, Vadasz, ...)
  - Our local strong pool (mlhybrid, structured, lb1224, ...)
  - Our main agent
Side-by-side avg behavioral features → see if our training data matches real distribution.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return [json.loads(l) for l in open(ROOT / path)]


KAGGLE = load("data/kaggle_replays_behaviors.jsonl")
LOCAL = load("data/opp_behaviors.jsonl")

FEATURES = [
    "launch_rate", "idle_rate", "avg_ships", "p90_ships", "ship_gini",
    "distinct_origin_frac", "early_planet_gain", "early_ship_growth",
    "avg_fleet_ratio", "centrality_drift", "aggression", "n_launches",
]


def stats(rows, label):
    if not rows:
        return None
    out = {"label": label, "n": len(rows)}
    for k in FEATURES:
        vals = [r[k] for r in rows]
        out[k] = sum(vals) / len(vals)
    out["win_rate"] = sum(1 for r in rows if r["is_winner"]) / len(rows)
    return out


# Kaggle: per-player
groups = []
for name in ["typeIIIfairy", "bowwowforeach", "Vadasz", "Isaiah @ Tufa Labs",
             "Ebi", "Shun_PI"]:
    rows = [r for r in KAGGLE if r["agent_name"] == name]
    s = stats(rows, f"K:{name}")
    if s: groups.append(s)

# Local strong pool
for name in ["mlhybrid", "lb1224", "structured", "proto1000",
             "orbitbotnext", "sundodge", "peak1103"]:
    rows = [r for r in LOCAL if r["agent_name"] == name]
    s = stats(rows, f"L:{name}")
    if s: groups.append(s)

# Local main
rows = [r for r in LOCAL if r["agent_name"] == "main"]
s = stats(rows, "L:main")
if s: groups.append(s)

# Pretty print
print(f"{'agent':<24}{'n':>5}", end="")
for k in FEATURES:
    print(f"{k[:9]:>10}", end="")
print(f"{'WR%':>7}")
print("-" * (24+5+10*len(FEATURES)+7))
for g in groups:
    print(f"{g['label']:<24}{g['n']:>5}", end="")
    for k in FEATURES:
        print(f"{g[k]:>10.2f}", end="")
    print(f"{g['win_rate']*100:>6.0f}%")

print()
print("=" * 80)
print("KEY DELTAS — top Kaggle player vs our strong pool")
print("=" * 80)
top = next(g for g in groups if g["label"] == "K:typeIIIfairy")
mainrow = next(g for g in groups if g["label"] == "L:main")
local_strong = [g for g in groups if g["label"].startswith("L:") and g["label"] != "L:main"]
strong_avg = {k: sum(g[k] for g in local_strong)/len(local_strong) for k in FEATURES}

print(f"\n{'feature':<24}{'typeIIIfairy':>14}{'L:strong avg':>14}{'L:main':>14}  {'gap?'}")
for k in FEATURES:
    t = top[k]; sa = strong_avg[k]; mn = mainrow[k]
    delta_ts = t - sa
    rel = abs(delta_ts) / max(0.01, abs(sa))
    mark = " ⚠️" if rel > 0.3 else ""
    print(f"  {k:<22}{t:>14.3f}{sa:>14.3f}{mn:>14.3f}{mark}")

print()
print("=" * 80)
print("Opp who BEATS typeIIIfairy: bowwowforeach (65% win rate)")
print("=" * 80)
bow = next(g for g in groups if g["label"] == "K:bowwowforeach")
print(f"\n{'feature':<24}{'bowwowforeach':>14}{'typeIIIfairy':>14}{'delta':>10}")
for k in FEATURES:
    print(f"  {k:<22}{bow[k]:>14.3f}{top[k]:>14.3f}{bow[k]-top[k]:>10.3f}")
