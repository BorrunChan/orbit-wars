"""
Deep-dive on 4P Kaggle replays (47 mid-tier games).

Goals:
  1. Identify the strongest 4P players in this sample (highest win rate)
  2. Compare their behavior vs typeIIIfairy (2P top) vs our local 4P data
  3. Verify map distribution matches our local sample
  4. Find what separates winners from losers in 4P
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(name):
    return [json.loads(l) for l in open(ROOT / "data" / name)]


KAG_4P = load("kaggle_4p_behaviors.jsonl")
KAG_2P = load("kaggle_2p_behaviors.jsonl")
KAG_4P_MAPS = load("kaggle_4p_maps.jsonl")
KAG_2P_MAPS = load("kaggle_2p_maps.jsonl")
LOCAL_4P = load("opp_behaviors.jsonl")
LOCAL_MAPS = load("map_signatures.jsonl")

FEATURES = [
    "launch_rate", "idle_rate", "avg_ships", "p90_ships", "ship_gini",
    "distinct_origin_frac", "early_planet_gain", "early_ship_growth",
    "avg_fleet_ratio", "centrality_drift", "aggression", "n_launches",
]


def stat_block(rows):
    if not rows: return None
    out = {"n": len(rows)}
    for k in FEATURES:
        vs = [r[k] for r in rows]
        out[k] = sum(vs)/len(vs)
    out["win_rate"] = sum(1 for r in rows if r["is_winner"]) / len(rows)
    return out


# ----------------------------------------------------------------------
# Section 1: Top 4P players (ranked by win rate in this sample)
# ----------------------------------------------------------------------
print("=" * 78)
print("🏆 STRONGEST KAGGLE 4P PLAYERS (this 47-game sample)")
print("=" * 78)
by_agent = defaultdict(list)
for r in KAG_4P:
    by_agent[r["agent_name"]].append(r)

ranked = []
for name, rows in by_agent.items():
    if len(rows) < 4: continue   # need min sample
    wr = sum(1 for r in rows if r["is_winner"]) / len(rows)
    ranked.append((name, len(rows), wr))
ranked.sort(key=lambda x: -x[2])

print(f"{'agent':<28}{'games':>7}{'win%':>7}{'expected':>10}")
print("-" * 52)
for name, n, wr in ranked:
    # 4P expected = 25%; flag if much higher/lower
    flag = "  ⭐" if wr > 0.35 else ("  ⬇️" if wr < 0.15 else "")
    print(f"  {name:<26}{n:>7}{wr*100:>6.0f}%{'25%':>10}{flag}")

# ----------------------------------------------------------------------
# Section 2: Behavior comparison — top 4P kaggle vs others vs local
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("📊 4P BEHAVIOR COMPARISON")
print("=" * 78)

# Top 4P kaggle: anyone with ≥35% win rate AND ≥5 games
top_4p_names = {name for name, n, wr in ranked if wr >= 0.35 and n >= 5}
top_4p_rows = [r for r in KAG_4P if r["agent_name"] in top_4p_names]
mid_4p_rows = [r for r in KAG_4P if r["agent_name"] not in top_4p_names]

# typeIIIfairy 2P
typeIII_rows = [r for r in KAG_2P if r["agent_name"] == "typeIIIfairy"]
# Local 4P opps (all 7 strong pool combined)
strong = {"mlhybrid", "lb1224", "structured", "proto1000",
          "orbitbotnext", "sundodge", "peak1103"}
local_strong = [r for r in LOCAL_4P if r["agent_name"] in strong]
local_main = [r for r in LOCAL_4P if r["agent_name"] == "main"]
# Split local main by winner/loser status to see if winners look different
local_main_win = [r for r in local_main if r["is_winner"]]
local_main_lose = [r for r in local_main if not r["is_winner"]]
# Best local agent: proto1000
local_proto = [r for r in LOCAL_4P if r["agent_name"] == "proto1000"]

groups = [
    ("Kaggle 4P TOP " + str(sorted(top_4p_names)), top_4p_rows),
    ("Kaggle 4P MID/LOW", mid_4p_rows),
    ("Kaggle 2P typeIIIfairy", typeIII_rows),
    ("Local 4P strong pool", local_strong),
    ("Local 4P proto1000 (69%WR)", local_proto),
    ("Local 4P main (winners)", local_main_win),
    ("Local 4P main (losers)", local_main_lose),
]

print(f"\n{'group':<48}{'n':>5}", end="")
for k in FEATURES:
    print(f"{k[:8]:>9}", end="")
print(f"{'WR%':>7}")
print("-" * (48+5+9*len(FEATURES)+7))
for label, rows in groups:
    if not rows: continue
    s = stat_block(rows)
    print(f"  {label:<46}{s['n']:>5}", end="")
    for k in FEATURES:
        print(f"{s[k]:>9.2f}", end="")
    print(f"{s['win_rate']*100:>6.0f}%")

# ----------------------------------------------------------------------
# Section 3: Map distribution comparison
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("🗺  MAP DISTRIBUTION: Kaggle 4P vs Local sampling")
print("=" * 78)

def map_dist(maps, label):
    avs = [m["angular_velocity"] for m in maps]
    nps = Counter(m["n_planets"] for m in maps)
    print(f"\n{label}:  N={len(maps)}")
    print(f"  angular_velocity:  min={min(avs):.4f} max={max(avs):.4f} "
          f"mean={sum(avs)/len(avs):.4f}")
    print(f"  n_planets distrib (count → fraction):")
    for n in sorted(nps):
        pct = nps[n]/len(maps)*100
        bar = "█" * int(pct/2)
        print(f"    {n:>3}  {nps[n]:>4}  {pct:>5.1f}%  {bar}")

map_dist(KAG_4P_MAPS, "Kaggle 4P (47 real games)")
map_dist(KAG_2P_MAPS, "Kaggle 2P (95 real games)")
map_dist(LOCAL_MAPS, "Local sample (1500 seeds)")

# ----------------------------------------------------------------------
# Section 4: Winner vs Loser within 4P
# ----------------------------------------------------------------------
print()
print("=" * 78)
print("🎯 WINNER vs LOSER (4P Kaggle)")
print("=" * 78)
winners = [r for r in KAG_4P if r["is_winner"]]
losers = [r for r in KAG_4P if not r["is_winner"]]
w_s = stat_block(winners); l_s = stat_block(losers)
print(f"\n{'feature':<24}{'winner':>10}{'loser':>10}{'gap':>10}{'gap%':>8}")
for k in FEATURES:
    g = w_s[k] - l_s[k]
    pct = g / max(0.001, abs(l_s[k])) * 100
    flag = " ⭐" if abs(pct) > 30 else ""
    print(f"  {k:<22}{w_s[k]:>10.2f}{l_s[k]:>10.2f}{g:>10.2f}{pct:>7.0f}%{flag}")
