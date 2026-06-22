"""Loss analysis restricted to reyhan_1239 (ref 53704199) and LATER submissions.

The full corpus mixes weak historical agents (v98/v124/v133); this filters to the
1239+ episode set (from /tmp/post1239_eps.json) so the opponent-style x rating-tier
win-rates reflect ONLY our current strong agents.
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent.parent
keep = set(json.load(open("/tmp/post1239_eps.json"))["allids"])
keep = {int(x) for x in keep}

styles = {}
for l in open(R / "data/opponent_styles.jsonl"):
    o = json.loads(l); styles[o["opp_name"]] = o

idx = [json.loads(l) for l in open(R / "data/replay_index.jsonl")]
idx = [g for g in idx if g.get("episode_id") in keep]

def tier(s):
    if s is None: return "no_lb"
    return ">1400" if s > 1400 else "1100-1400" if s > 1100 else "<1100"

# overall
for fmt in ("2P", "4P"):
    gs = [g for g in idx if g.get("format") == fmt]
    if gs:
        w = sum(1 for g in gs if g["my_won"])
        print(f"{fmt}: {w}/{len(gs)} = {100*w/len(gs):.0f}%")

seg = defaultdict(lambda: [0, 0])
loss_rating = []
for g in idx:
    if "my_idx" not in g: continue
    opp = [n for i, n in enumerate(g["team_names"]) if i != g["my_idx"]]
    wi = g.get("winner_idx")
    key_opp = g["team_names"][wi] if (not g["my_won"] and wi is not None) else (opp[0] if opp else None)
    if not key_opp or key_opp not in styles: continue
    o = styles[key_opp]; a = o["archetype"]; t = tier(o.get("lb_score"))
    seg[(a, t)][0] += 1; seg[(a, t)][1] += 1 if g["my_won"] else 0
    if not g["my_won"] and o.get("lb_score") is not None:
        loss_rating.append(o["lb_score"])

print(f"\n=== 1239+ only: win-rate by archetype x rating tier (games>=5) ===")
print(f"{'archetype':20} {'tier':>11} {'games':>6} {'ourWR':>6}")
for (a, t), (n, w) in sorted(seg.items(), key=lambda x: -x[1][0]):
    if n >= 5:
        print(f"{a:20} {t:>11} {n:>6} {100*w/n:>5.0f}%")

if loss_rating:
    lr = sorted(loss_rating)
    print(f"\n=== who beats us (1239+), n={len(lr)} losses ===")
    print(f"  median={st.median(lr):.0f} mean={st.mean(lr):.0f}")
    print(f"  >1400: {100*sum(1 for x in lr if x>1400)/len(lr):.0f}%  "
          f"1100-1400: {100*sum(1 for x in lr if 1100<x<=1400)/len(lr):.0f}%  "
          f"<1100: {100*sum(1 for x in lr if x<=1100)/len(lr):.0f}%")
