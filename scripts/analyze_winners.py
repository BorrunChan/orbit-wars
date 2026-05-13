"""
Analyze winner trajectories collected by collect_data.py.

Computes per-step features and aggregates patterns:
  - When do winners launch their first fleet?
  - Fleet size as fraction of source planet's ships
  - Multi-fleet turn frequency
  - Target priority (neutral vs enemy, near vs far)
  - Game-phase shifts (early / mid / late behavior)

Also compares winners across agents — lb1224 vs proto1000 vs structured vs main.

Usage:
    .venv/bin/python scripts/analyze_winners.py
"""

import json, math, statistics
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "winner_traces.jsonl"


def first_launch_turn(traj):
    """First step where the player launched any fleet."""
    for i, t in enumerate(traj):
        if t["action"]:
            return i
    return None


def total_ships_at_step(state):
    return sum(p["ships"] for p in state["my_planets"]) + \
           sum(f["ships"] for f in state["my_fleets"])


def planet_count(state, kind):
    return len(state[kind])


def actions_stats(traj):
    """Per-trajectory aggregates."""
    n_turns_with_action = 0
    total_actions = 0
    fleet_sizes = []
    ships_sent_fractions = []  # ratio of action ships to source planet ships
    ships_total_planet_at_action = []  # source planet ship count at decision time
    target_distances = []
    target_owners = Counter()
    target_prods = []
    multi_fleet_turns = 0

    for i, t in enumerate(traj):
        action = t["action"]
        state = t["state"]
        if not action:
            continue
        n_turns_with_action += 1
        total_actions += len(action)
        if len(action) >= 2:
            multi_fleet_turns += 1

        # Build planet lookup at this step
        all_planets = state["my_planets"] + state["enemy_planets"] + state["neutral_planets"]
        pl_lookup = {p["id"]: p for p in all_planets}
        my_pl = {p["id"]: p for p in state["my_planets"]}

        for mv in action:
            if not isinstance(mv, list) or len(mv) != 3:
                continue
            from_id, angle, ships = mv
            fleet_sizes.append(int(ships))
            src = my_pl.get(from_id)
            if src is not None:
                # source's ships at the moment of decision (env hasn't applied launch yet)
                src_ships = src["ships"]
                ships_total_planet_at_action.append(src_ships)
                if src_ships > 0:
                    ships_sent_fractions.append(int(ships) / src_ships)
            # Approximate target: ray cast — find nearest planet within an angle band
            # (rough; just take the closest planet not owned by self within 5 of ray dir)
            if src is not None:
                dx = math.cos(angle)
                dy = math.sin(angle)
                best = None; best_d = float("inf")
                for p in all_planets:
                    if p["id"] == from_id:
                        continue
                    rx = p["x"] - src["x"]
                    ry = p["y"] - src["y"]
                    # Project on ray
                    proj = rx*dx + ry*dy
                    if proj <= 0:
                        continue
                    perp = abs(rx*(-dy) + ry*dx)
                    if perp < 5 and proj < best_d:
                        best_d = proj; best = p
                if best is not None:
                    target_distances.append(best_d)
                    if "owner" in best:
                        target_owners["enemy"] += 1
                    elif best in state["neutral_planets"]:
                        target_owners["neutral"] += 1
                    else:
                        # likely own planet — reinforcement
                        target_owners["self"] += 1
                    target_prods.append(best["prod"])

    return {
        "n_turns_action": n_turns_with_action,
        "total_actions": total_actions,
        "fleet_sizes": fleet_sizes,
        "ships_sent_fractions": ships_sent_fractions,
        "src_ships_at_action": ships_total_planet_at_action,
        "target_distances": target_distances,
        "target_owners": dict(target_owners),
        "target_prods": target_prods,
        "multi_fleet_turns": multi_fleet_turns,
    }


def percentiles(vals, ps=(10, 25, 50, 75, 90)):
    if not vals:
        return {p: None for p in ps}
    sv = sorted(vals)
    out = {}
    for p in ps:
        idx = int(p / 100 * (len(sv) - 1))
        out[p] = sv[idx]
    return out


def main():
    if not DATA.exists():
        print(f"No data at {DATA}. Run collect_data.py first.")
        return

    # Load
    trajectories = []
    with open(DATA) as f:
        for line in f:
            trajectories.append(json.loads(line))
    print(f"Loaded {len(trajectories)} winner trajectories")

    # Group by agent (the winning agent)
    by_agent = defaultdict(list)
    for tr in trajectories:
        by_agent[tr["winner_agent"]].append(tr)

    print(f"\nWinner counts by agent:")
    for k, v in sorted(by_agent.items(), key=lambda kv: -len(kv[1])):
        print(f"  {k:<35}  {len(v):>4} wins")

    # Aggregate per agent
    print(f"\n{'='*100}")
    print(f"Per-agent winning-style aggregates:")
    print(f"{'-'*100}")
    cols = ("first_launch", "actions/turn", "ships/turn", "fleet_size",
            "src_ships", "fleet_frac_of_src", "tgt_owners", "tgt_prod")
    header = f"  {'agent':<32}"
    for c in cols:
        header += f"  {c:<14}"
    print(header)

    for agent, trs in sorted(by_agent.items(), key=lambda kv: -len(kv[1])):
        # Collect stats
        first_launches = []
        all_fleet_sizes = []
        all_fractions = []
        all_src_ships = []
        all_distances = []
        all_prods = []
        target_counter = Counter()
        for tr in trs:
            traj = tr["traj"]
            fl = first_launch_turn(traj)
            if fl is not None:
                first_launches.append(fl)
            s = actions_stats(traj)
            all_fleet_sizes.extend(s["fleet_sizes"])
            all_fractions.extend(s["ships_sent_fractions"])
            all_src_ships.extend(s["src_ships_at_action"])
            all_distances.extend(s["target_distances"])
            all_prods.extend(s["target_prods"])
            for k, v in s["target_owners"].items():
                target_counter[k] += v

        total_turns = sum(len(tr["traj"]) for tr in trs)
        total_actions = sum(actions_stats(tr["traj"])["total_actions"] for tr in trs)
        total_ships_sent = sum(all_fleet_sizes)

        def fmt(v, dec=1):
            if v is None: return "  -  "
            return f"{v:.{dec}f}"

        row = f"  {agent[-32:]:<32}"
        row += f"  {fmt(statistics.median(first_launches) if first_launches else None):<14}"
        row += f"  {fmt(total_actions/max(1,total_turns), 2):<14}"
        row += f"  {fmt(total_ships_sent/max(1,total_turns), 1):<14}"
        row += f"  {fmt(statistics.median(all_fleet_sizes) if all_fleet_sizes else None):<14}"
        row += f"  {fmt(statistics.median(all_src_ships) if all_src_ships else None):<14}"
        row += f"  {fmt(statistics.median(all_fractions) if all_fractions else None, 2):<14}"
        # target distribution
        tot = sum(target_counter.values()) or 1
        tgt_str = f"N{target_counter.get('neutral',0)*100//tot}/E{target_counter.get('enemy',0)*100//tot}"
        row += f"  {tgt_str:<14}"
        row += f"  {fmt(statistics.median(all_prods) if all_prods else None):<14}"
        print(row)

    # Phase analysis
    print(f"\n{'='*100}")
    print(f"Phase analysis (per agent):")
    print(f"{'-'*100}")
    phases = [("early", lambda s: s < 30), ("mid", lambda s: 30 <= s < 150), ("late", lambda s: s >= 150)]
    for agent, trs in sorted(by_agent.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {agent}:")
        for pname, pred in phases:
            actions_in_phase = 0
            turns_in_phase = 0
            fleet_sizes = []
            fractions = []
            for tr in trs:
                for t in tr["traj"]:
                    if not pred(t["state"]["step"]):
                        continue
                    turns_in_phase += 1
                    if t["action"]:
                        actions_in_phase += 1
                        my_pl = {p["id"]: p for p in t["state"]["my_planets"]}
                        for mv in t["action"]:
                            if isinstance(mv, list) and len(mv) == 3:
                                fleet_sizes.append(int(mv[2]))
                                src = my_pl.get(mv[0])
                                if src and src["ships"] > 0:
                                    fractions.append(int(mv[2]) / src["ships"])
            if turns_in_phase == 0:
                continue
            launch_rate = actions_in_phase / turns_in_phase
            med_size = statistics.median(fleet_sizes) if fleet_sizes else None
            med_frac = statistics.median(fractions) if fractions else None
            print(f"    {pname:<8}  turns={turns_in_phase:>5}  launch_rate={launch_rate:.0%}  "
                  f"med_fleet={med_size}  med_frac_of_src={med_frac:.2f}" if med_frac else
                  f"    {pname:<8}  turns={turns_in_phase:>5}  launch_rate={launch_rate:.0%}  "
                  f"med_fleet={med_size}")


if __name__ == "__main__":
    main()
