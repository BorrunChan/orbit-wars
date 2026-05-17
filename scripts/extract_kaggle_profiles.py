"""
Extract behavioral profiles from Kaggle replays for opponent modeling.

For each (replay, player) pair, computes:
  - launch_rate: avg launches per turn
  - avg_fleet, std_fleet: ship size distribution
  - target_enemy_rate, target_neutral_rate
  - early_density, mid_density, late_density: launches by phase
  - close_target_bias: prefer near vs far targets

Output: data/kaggle_opp_profiles.json (list of dicts)
"""
import json, math, os, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIRS = [Path("/Users/bor/Downloads"), ROOT / "replays", ROOT / "replays" / "4p"]
OUT = ROOT / "data" / "kaggle_opp_profiles.json"

CENTER = 50.0


def find_planet(planets, pid):
    for p in planets:
        if p[0] == pid:
            return p
    return None


def extract_profile(ep, slot, name):
    """Extract behavioral profile for player at slot."""
    steps = ep["steps"]
    n_steps = len(steps) - 1
    if n_steps < 20:
        return None
    
    n_p = len(steps[0])
    launches = []  # (step, from_pid, angle, ships, target_owner, target_dist)
    
    for s in range(n_steps):
        obs = steps[s][0].get("observation", {})
        planets = obs.get("planets") or []
        actions = steps[s][slot].get("action") or []
        for a in actions:
            if not isinstance(a, list) or len(a) < 3:
                continue
            from_pid, angle, ships = a[0], a[1], a[2]
            src = find_planet(planets, from_pid)
            if src is None:
                continue
            # Find target = closest opp/neutral in direction of angle (approximation)
            best_tgt = None; best_score = float('inf')
            for tgt in planets:
                if tgt[0] == from_pid: continue
                if tgt[1] == slot: continue  # not own
                dx = tgt[2] - src[2]; dy = tgt[3] - src[3]
                dist = math.hypot(dx, dy)
                tgt_angle = math.atan2(dy, dx)
                ang_diff = abs(((tgt_angle - angle + math.pi) % (2*math.pi)) - math.pi)
                # Score by angle proximity (closest direction match)
                score = ang_diff * 30 + dist * 0.5
                if score < best_score:
                    best_score = score; best_tgt = tgt
            if best_tgt is not None:
                tgt_dist = math.hypot(best_tgt[2]-src[2], best_tgt[3]-src[3])
                tgt_owner = best_tgt[1]  # -1 = neutral, else player idx
                launches.append((s, from_pid, angle, ships, tgt_owner, tgt_dist))
    
    if len(launches) < 5:
        return None
    
    # Compute features
    fleet_sizes = [l[3] for l in launches]
    target_owners = [l[4] for l in launches]
    target_dists = [l[5] for l in launches]
    
    # Phase split
    early = [l for l in launches if l[0] < n_steps * 0.33]
    mid   = [l for l in launches if n_steps * 0.33 <= l[0] < n_steps * 0.66]
    late  = [l for l in launches if l[0] >= n_steps * 0.66]
    
    early_steps = max(1, int(n_steps * 0.33))
    mid_steps = max(1, int(n_steps * 0.33))
    late_steps = max(1, n_steps - early_steps - mid_steps)
    
    return {
        "name": name,
        "n_steps": n_steps,
        "n_launches": len(launches),
        "launch_rate": len(launches) / n_steps,
        "avg_fleet": statistics.mean(fleet_sizes),
        "std_fleet": statistics.stdev(fleet_sizes) if len(fleet_sizes) > 1 else 0,
        "median_fleet": statistics.median(fleet_sizes),
        "p90_fleet": sorted(fleet_sizes)[int(len(fleet_sizes)*0.9)],
        "p_target_neutral": sum(1 for o in target_owners if o == -1) / len(target_owners),
        "p_target_enemy": sum(1 for o in target_owners if o != -1 and o != slot) / len(target_owners),
        "avg_target_dist": statistics.mean(target_dists),
        "early_density": len(early) / early_steps,
        "mid_density": len(mid) / mid_steps,
        "late_density": len(late) / late_steps,
        "num_players": n_p,
    }


def main():
    profiles = []
    seen_files = set()
    for d in REPLAY_DIRS:
        if not d.exists():
            continue
        for f in d.glob("episode-*.json"):
            if f.name in seen_files:
                continue
            seen_files.add(f.name)
            try:
                ep = json.load(open(f))
                teams = ep["info"].get("TeamNames", [])
                n_p = len(ep["steps"][0])
                for slot in range(n_p):
                    name = teams[slot] if slot < len(teams) else f"slot{slot}"
                    if name == "Borrun":  # skip our own
                        continue
                    p = extract_profile(ep, slot, name)
                    if p is not None:
                        p["replay_file"] = f.name
                        profiles.append(p)
            except Exception as e:
                print(f"ERR {f.name}: {e}")
    
    print(f"Extracted {len(profiles)} opp profiles from {len(seen_files)} replays")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(profiles, indent=2))
    print(f"Saved to {OUT}")
    
    # Summary
    print("\n=== Profile summary ===")
    print(f"{'name':<30}{'launches':>10}{'avg_flt':>10}{'launch_rate':>15}")
    for p in sorted(profiles, key=lambda x: -x["n_launches"])[:20]:
        print(f"{p['name'][:28]:<30}{p['n_launches']:>10}{p['avg_fleet']:>10.1f}{p['launch_rate']:>15.3f}")


if __name__ == "__main__":
    main()
