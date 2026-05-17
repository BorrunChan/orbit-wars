"""
Parse downloaded Kaggle replays → extract:
  - Map signature (geometric, per replay)
  - Per-player behavioral fingerprint (per (replay, player_slot))
  - Outcome + agent name

Outputs:
  data/kaggle_replays_maps.jsonl      one row per replay
  data/kaggle_replays_behaviors.jsonl one row per (replay, player_slot)

Optionally compares against our local sample distribution + behavior clusters.

Usage:
  .venv/bin/python scripts/analyze_kaggle_replays.py [replays_dir]
"""
import argparse, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "replays"
OUT_MAPS = ROOT / "data" / "kaggle_replays_maps.jsonl"
OUT_BEH = ROOT / "data" / "kaggle_replays_behaviors.jsonl"

CENTER = 50.0
OBS_WINDOW = 30


def gini(xs):
    if not xs: return 0.0
    xs = sorted(xs); n = len(xs); s = sum(xs)
    if s < 1e-9: return 0.0
    cum = sum((i+1)*x for i, x in enumerate(xs))
    return (2*cum)/(n*s) - (n+1)/n


def map_signature(obs0, seed):
    planets = obs0.get("planets") or []
    av = obs0.get("angular_velocity", 0)
    n_p = len(planets)
    if n_p == 0:
        return None
    radii = [math.hypot(p[2]-CENTER, p[3]-CENTER) for p in planets]
    owners = [p[1] for p in planets]
    n_owned = sum(1 for o in owners if o >= 0)
    n_neutral = sum(1 for o in owners if o == -1)
    prods = [p[6] for p in planets]
    ships = [p[5] for p in planets]
    planet_r = [p[4] for p in planets]

    # min pairwise planet dist
    min_dist = float("inf")
    for i in range(n_p):
        for j in range(i+1, n_p):
            d = math.hypot(planets[i][2]-planets[j][2], planets[i][3]-planets[j][3])
            if d < min_dist: min_dist = d

    n_rotating = sum(1 for p, r in zip(planets, radii) if r + p[4] < 50.0)
    n_inner = sum(1 for r in radii if r < 25)
    n_outer = n_p - n_inner

    # sun-blocking pairs (same heuristic as extract_map_signatures.py)
    n_sun_pair = 0
    for i in range(n_p):
        for j in range(i+1, n_p):
            xi, yi = planets[i][2]-CENTER, planets[i][3]-CENTER
            xj, yj = planets[j][2]-CENTER, planets[j][3]-CENTER
            mi = math.hypot(xi, yi); mj = math.hypot(xj, yj)
            if mi*mj < 1e-9: continue
            cos_a = (xi*xj + yi*yj) / (mi*mj)
            if cos_a < -0.85 and mi < 35 and mj < 35:
                n_sun_pair += 1

    return {
        "seed": seed,
        "angular_velocity": round(av, 5),
        "n_planets": n_p,
        "n_planet_groups": n_p // 4,
        "n_rotating": n_rotating,
        "rotating_frac": round(n_rotating / n_p, 3),
        "min_planet_dist": round(min_dist, 2),
        "avg_orbital_radius": round(sum(radii)/n_p, 2),
        "std_orbital_radius": round((sum((r - sum(radii)/n_p)**2 for r in radii)/n_p)**0.5, 2),
        "n_inner_planets": n_inner,
        "n_outer_planets": n_outer,
        "n_sun_block_pairs": n_sun_pair,
        "total_prod": sum(prods),
        "avg_prod": round(sum(prods)/n_p, 2),
        "avg_planet_radius": round(sum(planet_r)/n_p, 3),
        "avg_initial_ships_per_planet": round(sum(ships)/n_p, 2),
        "n_owned_at_start": n_owned,  # = num_players in this format
        "n_neutral_at_start": n_neutral,
    }


def behavior_features(steps, player_slot, num_players, window=OBS_WINDOW):
    """Compute behavioral features for player_slot over first `window` turns."""
    actions_list = []          # ships sizes
    distinct_origins = set()
    turns_with_action = 0
    n_turns = min(window, len(steps))
    for ti in range(n_turns):
        step = steps[ti][player_slot]
        a_list = step.get("action") or []
        if a_list:
            turns_with_action += 1
        for a in a_list:
            if isinstance(a, list) and len(a) >= 3:
                distinct_origins.add(a[0])
                actions_list.append(a[2])

    launch_rate = len(actions_list) / max(1, n_turns)
    idle_rate = 1 - turns_with_action / max(1, n_turns)
    if actions_list:
        avg_ships = statistics.mean(actions_list)
        med_ships = statistics.median(actions_list)
        p90_ships = sorted(actions_list)[int(len(actions_list)*0.9)]
        ship_gini = gini(actions_list)
    else:
        avg_ships = med_ships = p90_ships = ship_gini = 0.0

    # Derive per-turn snapshots from obs
    def snapshot(step_idx):
        obs = steps[step_idx][player_slot].get("observation") or {}
        planets = obs.get("planets") or []
        fleets = obs.get("fleets") or []
        my_planets = sum(1 for p in planets if p[1] == player_slot)
        my_ships_pl = sum(p[5] for p in planets if p[1] == player_slot)
        my_fleet_ships = sum(f[6] for f in fleets if f[1] == player_slot)
        my_total = my_ships_pl + my_fleet_ships
        # centrality: avg dist of my planets to board center
        my_xy = [(p[2], p[3]) for p in planets if p[1] == player_slot]
        if my_xy:
            cx = sum(x for x,y in my_xy)/len(my_xy)
            cy = sum(y for x,y in my_xy)/len(my_xy)
            cent = math.hypot(cx-CENTER, cy-CENTER)
        else:
            cent = 0
        return {"planets": my_planets, "ships": my_total,
                "fleet": my_fleet_ships, "centrality": cent}

    s0 = snapshot(0)
    sN = snapshot(min(window-1, len(steps)-1))
    planet_gain = sN["planets"] - s0["planets"]
    ship_growth = sN["ships"] / max(1, s0["ships"])

    fleet_ratios = []
    for ti in range(5, n_turns):
        snap = snapshot(ti)
        if snap["ships"] > 0:
            fleet_ratios.append(snap["fleet"] / snap["ships"])
    avg_fleet_ratio = statistics.mean(fleet_ratios) if fleet_ratios else 0.0
    centrality_drift = sN["centrality"] - s0["centrality"]

    avg_per_planet = (sN["ships"] - sN["fleet"]) / max(1, sN["planets"])
    aggression = avg_ships / max(1.0, avg_per_planet) if avg_ships else 0.0

    return {
        "launch_rate": round(launch_rate, 3),
        "idle_rate": round(idle_rate, 3),
        "avg_ships": round(avg_ships, 2),
        "med_ships": round(med_ships, 2),
        "p90_ships": round(p90_ships, 2),
        "ship_gini": round(ship_gini, 3),
        "n_distinct_origins": len(distinct_origins),
        "distinct_origin_frac": round(len(distinct_origins) / max(1, sN["planets"]), 3),
        "early_planet_gain": planet_gain,
        "early_ship_growth": round(ship_growth, 3),
        "avg_fleet_ratio": round(avg_fleet_ratio, 3),
        "centrality_drift": round(centrality_drift, 2),
        "aggression": round(aggression, 3),
        "n_launches": len(actions_list),
    }


def parse_replay(path):
    ep = json.load(open(path))
    info = ep.get("info", {})
    seed = info.get("seed", 0)
    team_names = info.get("TeamNames", [])
    steps = ep["steps"]
    n_players = len(steps[0])
    rewards = ep.get("rewards", [0]*n_players)
    statuses = ep.get("statuses", ["UNKNOWN"]*n_players)

    if not steps[0][0].get("observation", {}).get("planets"):
        return None, None  # incomplete

    obs0 = steps[0][0]["observation"]
    map_sig = map_signature(obs0, seed)
    if not map_sig:
        return None, None
    map_sig["episode_id"] = info.get("EpisodeId", path.stem)
    map_sig["n_players"] = n_players
    map_sig["total_steps"] = len(steps)
    map_sig["team_names"] = team_names

    # Winner = max reward; in case of tie/draw, no winner
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r and r > 0]
    winner = winners[0] if len(winners) == 1 else -1

    behaviors = []
    for p in range(n_players):
        beh = behavior_features(steps, p, n_players)
        beh.update({
            "episode_id": map_sig["episode_id"],
            "seed": seed,
            "player_slot": p,
            "agent_name": team_names[p] if p < len(team_names) else f"player_{p}",
            "is_winner": p == winner,
            "reward": rewards[p],
            "status": statuses[p],
            "num_players": n_players,
            "total_steps": len(steps),
        })
        behaviors.append(beh)

    return map_sig, behaviors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=str(DEFAULT_DIR))
    args = ap.parse_args()

    files = sorted(Path(args.dir).glob("episode-*.json"))
    print(f"Found {len(files)} replays in {args.dir}")

    OUT_MAPS.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_err = 0
    all_maps, all_beh = [], []
    with open(OUT_MAPS, "w") as fm, open(OUT_BEH, "w") as fb:
        for f in files:
            try:
                m, bs = parse_replay(f)
                if m is None:
                    n_err += 1; continue
                fm.write(json.dumps(m) + "\n"); all_maps.append(m)
                for b in bs:
                    fb.write(json.dumps(b) + "\n"); all_beh.append(b)
                n_ok += 1
            except Exception as e:
                print(f"ERR {f.name}: {e}"); n_err += 1
    print(f"\nParsed {n_ok} replays ({n_err} errors)")
    print(f"  → {OUT_MAPS} ({len(all_maps)} maps)")
    print(f"  → {OUT_BEH} ({len(all_beh)} behavior rows)")
    return all_maps, all_beh


def summarize(all_maps, all_beh, focus_player="typeIIIfairy"):
    print("\n" + "="*70)
    print(f"📊 Summary — focus on '{focus_player}'")
    print("="*70)

    # Game format distribution
    fmt = Counter(m["n_players"] for m in all_maps)
    print(f"\nGame formats: {dict(fmt)}")

    # Map dim distributions
    avs = [m["angular_velocity"] for m in all_maps]
    nps = [m["n_planets"] for m in all_maps]
    print(f"angular_velocity:  min={min(avs):.4f} max={max(avs):.4f} mean={sum(avs)/len(avs):.4f}")
    print(f"n_planets distrib: {Counter(nps)}")

    # focus player W/L per opponent
    p_rows = [b for b in all_beh if b["agent_name"] == focus_player]
    print(f"\n{focus_player}: {len(p_rows)} games")
    print(f"  wins:   {sum(1 for r in p_rows if r['is_winner'])}")
    print(f"  losses: {sum(1 for r in p_rows if not r['is_winner'] and r['reward'] != 0)}")

    # Vs whom?
    opp_results = defaultdict(lambda: [0, 0])  # wins, total
    for b in all_beh:
        if b["agent_name"] != focus_player:
            continue
        # Find opponents in same episode
        ep = b["episode_id"]
        others = [x["agent_name"] for x in all_beh
                  if x["episode_id"] == ep and x["agent_name"] != focus_player]
        for o in others:
            opp_results[o][1] += 1
            if b["is_winner"]:
                opp_results[o][0] += 1
    print(f"\n{focus_player}'s opponents (sorted by frequency):")
    print(f"  {'opponent':<28}  games  win%")
    for opp, (w, n) in sorted(opp_results.items(), key=lambda x: -x[1][1]):
        if n >= 2:
            print(f"  {opp:<28}  {n:>5}  {w/n*100:>4.0f}%")
    print(f"\n  (only opponents seen ≥2 times shown)")

    # focus player's behavioral fingerprint (averaged)
    if p_rows:
        print(f"\n{focus_player}'s avg behavior (first {OBS_WINDOW} turns):")
        for k in ["launch_rate", "idle_rate", "avg_ships", "p90_ships",
                  "early_planet_gain", "avg_fleet_ratio", "centrality_drift",
                  "aggression", "n_launches"]:
            vals = [r[k] for r in p_rows]
            avg = sum(vals)/len(vals)
            wins = [r[k] for r in p_rows if r["is_winner"]]
            losses = [r[k] for r in p_rows if not r["is_winner"]]
            w_avg = sum(wins)/max(1, len(wins))
            l_avg = sum(losses)/max(1, len(losses))
            print(f"  {k:<22}  avg={avg:>7.3f}  win={w_avg:>7.3f}  lose={l_avg:>7.3f}")


if __name__ == "__main__":
    all_maps, all_beh = main()
    summarize(all_maps, all_beh)
