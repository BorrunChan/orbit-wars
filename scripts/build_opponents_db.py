"""
Build data/opponents_db.jsonl + data/opponents_db.md from indexed replays.

For each opponent (TeamName), aggregate over all replays containing our team:
  - encounter_n, my_wins, my_wr
  - per-format breakdown (2P / 4P-as-rank1 / 4P-overall)
  - launch_rate: mean actions/turn submitted by this opponent
  - actual_fleet_rate: mean fleets actually created/turn (via next_fleet_id delta)
  - avg_fleet_size, first_launch_step, P_neutral_target
  - sample_replays (up to 3 episode IDs)
  - archetype tag (high_volume / moderate / structured / low_activity) by launch rate

Usage:
  .venv/bin/python scripts/build_opponents_db.py
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / 'data' / 'replay_index.jsonl'
OUT_JSONL = ROOT / 'data' / 'opponents_db.jsonl'
OUT_MD = ROOT / 'data' / 'opponents_db.md'


def per_replay_opp_metrics(replay_data, opp_slot):
    """For one replay, compute opp's per-game metrics."""
    steps = replay_data.get('steps', [])
    submitted = 0
    first_launch = None
    fleet_sizes = []
    neutral_target_count = 0
    enemy_target_count = 0
    last_nfid = 0
    for step_idx, step_data in enumerate(steps):
        if opp_slot >= len(step_data):
            continue
        slot = step_data[opp_slot]
        action = slot.get('action') or []
        if isinstance(action, list):
            for a in action:
                if isinstance(a, list) and len(a) >= 3:
                    submitted += 1
                    fleet_sizes.append(a[2])
                    if first_launch is None:
                        first_launch = step_idx
        # planets info to classify neutral vs enemy targets
        obs = slot.get('observation') or {}
        nfid = obs.get('next_fleet_id') or 0
        last_nfid = max(last_nfid, nfid)
    actual_launches = last_nfid  # approximate; better: delta vs other slots
    total_steps = max(1, len(steps))
    return {
        "submitted_actions": submitted,
        "submit_rate": submitted / total_steps,
        "avg_fleet_size": (sum(fleet_sizes) / len(fleet_sizes)
                           if fleet_sizes else 0),
        "first_launch_step": first_launch,
        "steps": total_steps,
        "approx_actual_nfid_at_end": last_nfid,
    }


def main():
    # Load index
    index_records = []
    with open(INDEX) as f:
        for line in f:
            index_records.append(json.loads(line))

    # opp_name -> list of per-replay metrics + WR info
    agg = defaultdict(lambda: {
        "encounters": 0,
        "my_wins": 0,
        "encounters_2p": 0, "my_wins_2p": 0,
        "encounters_4p": 0, "my_wins_4p": 0,
        "submit_rates": [], "fleet_sizes": [], "first_launches": [],
        "sample_replays": [],
    })

    for idx_rec in index_records:
        if idx_rec.get('my_idx', -1) < 0:
            continue  # Only replays with our team
        my_idx = idx_rec['my_idx']
        teams = idx_rec.get('team_names', [])
        fmt = idx_rec['format']
        my_won = idx_rec.get('my_won', False)

        # Load actual replay for action analysis
        replay_path = ROOT / idx_rec['path']
        try:
            with open(replay_path) as f:
                rep = json.load(f)
        except Exception:
            continue

        for opp_slot, opp_name in enumerate(teams):
            if opp_slot == my_idx:
                continue
            metrics = per_replay_opp_metrics(rep, opp_slot)
            a = agg[opp_name]
            a["encounters"] += 1
            a[f"encounters_{fmt.lower()}"] += 1
            if my_won:
                a["my_wins"] += 1
                a[f"my_wins_{fmt.lower()}"] += 1
            a["submit_rates"].append(metrics["submit_rate"])
            if metrics["avg_fleet_size"] > 0:
                a["fleet_sizes"].append(metrics["avg_fleet_size"])
            if metrics["first_launch_step"] is not None:
                a["first_launches"].append(metrics["first_launch_step"])
            if len(a["sample_replays"]) < 3:
                a["sample_replays"].append({
                    "episode_id": idx_rec["episode_id"],
                    "format": fmt,
                    "my_won": my_won,
                    "submit_rate": round(metrics["submit_rate"], 2),
                })

    # Build records
    records = []
    for name, a in agg.items():
        n = a["encounters"]
        rec = {
            "opp_name": name,
            "encounters": n,
            "my_wins": a["my_wins"],
            "my_wr": round(100 * a["my_wins"] / max(1, n), 1),
            "encounters_2p": a["encounters_2p"],
            "my_wr_2p": (round(100 * a["my_wins_2p"]
                                / max(1, a["encounters_2p"]), 1)
                          if a["encounters_2p"] else None),
            "encounters_4p": a["encounters_4p"],
            "my_wr_4p": (round(100 * a["my_wins_4p"]
                                / max(1, a["encounters_4p"]), 1)
                          if a["encounters_4p"] else None),
            "avg_submit_rate": round(
                sum(a["submit_rates"]) / max(1, len(a["submit_rates"])), 2),
            "max_submit_rate": round(max(a["submit_rates"] or [0]), 2),
            "avg_fleet_size": round(
                sum(a["fleet_sizes"]) / max(1, len(a["fleet_sizes"])), 1),
            "median_first_launch": (sorted(a["first_launches"])
                [len(a["first_launches"]) // 2]
                if a["first_launches"] else None),
            "sample_replays": a["sample_replays"],
        }
        # Archetype classification by avg_submit_rate (submitted actions/turn)
        r = rec["avg_submit_rate"]
        if r >= 4.0:
            rec["archetype"] = "high_volume_swarm"
        elif r >= 1.5:
            rec["archetype"] = "moderate"
        elif r >= 0.5:
            rec["archetype"] = "structured"
        else:
            rec["archetype"] = "low_activity"
        records.append(rec)

    # Sort by encounters desc
    records.sort(key=lambda r: r["encounters"], reverse=True)

    # Write JSONL
    OUT_JSONL.parent.mkdir(exist_ok=True)
    with open(OUT_JSONL, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"Wrote {len(records)} opponents → {OUT_JSONL}")

    # Write Markdown summary
    md = ["# Opponents Database",
          "",
          f"Built from {len(records)} unique opponents across "
          f"{sum(r['encounters'] for r in records)} encounters.",
          "",
          "## Archetype distribution",
          ""]
    arch_count = defaultdict(int)
    arch_encounters = defaultdict(int)
    arch_wins = defaultdict(int)
    for r in records:
        arch_count[r["archetype"]] += 1
        arch_encounters[r["archetype"]] += r["encounters"]
        arch_wins[r["archetype"]] += r["my_wins"]
    md.append("| archetype | n_opps | encounters | my_wr |")
    md.append("|---|---:|---:|---:|")
    for arch in ["high_volume_swarm", "moderate", "structured",
                  "low_activity"]:
        nopps = arch_count.get(arch, 0)
        ne = arch_encounters.get(arch, 0)
        nw = arch_wins.get(arch, 0)
        wr = round(100 * nw / max(1, ne), 1)
        md.append(f"| {arch} | {nopps} | {ne} | {nw}/{ne} = {wr}% |")
    md.append("")
    md.append("## Per-opponent table (sorted by encounters)")
    md.append("")
    md.append("| opp | n | wr | wr_2p | wr_4p | submit/t | fleet | "
              "1st_l | archetype |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in records:
        wr2 = f"{r['my_wr_2p']}%" if r['my_wr_2p'] is not None else "-"
        wr4 = f"{r['my_wr_4p']}%" if r['my_wr_4p'] is not None else "-"
        fl = r['median_first_launch'] if r['median_first_launch'] is not None else "-"
        md.append(f"| {r['opp_name'][:30]} | {r['encounters']} | "
                  f"{r['my_wr']}% | {wr2} | {wr4} | "
                  f"{r['avg_submit_rate']} | {r['avg_fleet_size']} | "
                  f"{fl} | {r['archetype']} |")
    md.append("")
    md.append("## High-volume swarmers (the proto1000 zone)")
    md.append("")
    hv = [r for r in records if r["archetype"] == "high_volume_swarm"]
    hv.sort(key=lambda r: r["avg_submit_rate"], reverse=True)
    md.append("| opp | n | wr | wr_2p | wr_4p | submit/t | max/t | fleet |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in hv[:50]:
        wr2 = f"{r['my_wr_2p']}%" if r['my_wr_2p'] is not None else "-"
        wr4 = f"{r['my_wr_4p']}%" if r['my_wr_4p'] is not None else "-"
        md.append(f"| {r['opp_name'][:30]} | {r['encounters']} | "
                  f"{r['my_wr']}% | {wr2} | {wr4} | "
                  f"{r['avg_submit_rate']} | "
                  f"{r['max_submit_rate']} | {r['avg_fleet_size']} |")
    md.append("")
    md.append("## Moderate swarmers (submit_rate 1.5-4.0)")
    md.append("")
    mod = [r for r in records if r["archetype"] == "moderate"
           and r["encounters"] >= 2]
    mod.sort(key=lambda r: r["avg_submit_rate"], reverse=True)
    md.append("| opp | n | wr | wr_2p | wr_4p | submit/t | fleet |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in mod[:30]:
        wr2 = f"{r['my_wr_2p']}%" if r['my_wr_2p'] is not None else "-"
        wr4 = f"{r['my_wr_4p']}%" if r['my_wr_4p'] is not None else "-"
        md.append(f"| {r['opp_name'][:30]} | {r['encounters']} | "
                  f"{r['my_wr']}% | {wr2} | {wr4} | "
                  f"{r['avg_submit_rate']} | {r['avg_fleet_size']} |")
    md.append("")
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote summary → {OUT_MD}")


if __name__ == '__main__':
    main()
