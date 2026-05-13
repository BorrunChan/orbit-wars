"""
Orbit Wars - Agent v2

Additions over v1:
  - Defensive reinforcement: detect incoming enemy fleets, predict net deficit
    per planet, and dispatch reinforcements that arrive in time.
  - Multi-source capture stacking for neutral targets: if no single mine can
    afford a juicy neutral, two mines coordinate (neutrals do not regrow, so
    staggered arrivals chip the garrison down).
  - Comet path lookup for arrival-time prediction (was already in v1 helper,
    now used more aggressively).
  - Slight capture buffer on contested (enemy-owned) targets.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6
EPISODE_STEPS = 500
ENEMY_CAPTURE_MARGIN = 2  # extra ships when attacking an enemy-owned target


def _speed(ships: int) -> float:
    s = max(1, ships)
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5)


def _predict_position(pid, current_pos, initial_by_id, angular_velocity,
                       step, future_step, comet_groups):
    for group in comet_groups:
        if pid in group["planet_ids"]:
            i = group["planet_ids"].index(pid)
            new_idx = group["path_index"] + (future_step - step)
            path = group["paths"][i]
            if 0 <= new_idx < len(path):
                return (path[new_idx][0], path[new_idx][1])
            return None
    init = initial_by_id.get(pid)
    if init is None:
        return current_pos
    dx, dy = init[2] - CENTER, init[3] - CENTER
    r = math.hypot(dx, dy)
    if r + init[4] >= ROTATION_RADIUS_LIMIT:
        return (init[2], init[3])
    a0 = math.atan2(dy, dx)
    a1 = a0 + angular_velocity * future_step
    return (CENTER + r * math.cos(a1), CENTER + r * math.sin(a1))


def _safe_launch_angle(src, dst, target_radius):
    sx, sy = src
    tx, ty = dst
    tdx, tdy = tx - sx, ty - sy
    tdist = math.hypot(tdx, tdy)
    if tdist < 1e-6:
        return 0.0
    tgt_angle = math.atan2(tdy, tdx)

    sdx, sdy = CENTER - sx, CENTER - sy
    sdist = math.hypot(sdx, sdy)
    R = SUN_RADIUS + SUN_BUFFER
    if sdist <= R:
        return tgt_angle
    sun_angle = math.atan2(sdy, sdx)
    sun_half = math.asin(min(1.0, R / sdist))

    if tdist <= target_radius:
        return tgt_angle
    tgt_half = math.asin(min(1.0, target_radius / tdist))
    diff = (tgt_angle - sun_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) >= sun_half + tgt_half:
        return tgt_angle
    if diff >= 0:
        chosen = sun_angle + sun_half + 1e-3
    else:
        chosen = sun_angle - sun_half - 1e-3
    delta = (chosen - tgt_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(delta) > tgt_half - 1e-3:
        return None
    return chosen


def _capture_cost(mine_x, mine_y, target, predicted_dst, available, step,
                   initial_by_id, angular_velocity, comet_groups, player):
    """Iterate to a self-consistent (ships, T, arrival_pos) for one mine."""
    tgt_owner = target.owner
    tgt_prod = target.production
    tgt_ships = target.ships
    s = max(1, tgt_ships + 1)
    last_T = None
    arrival_pos = predicted_dst
    for _ in range(6):
        dx = arrival_pos[0] - mine_x
        dy = arrival_pos[1] - mine_y
        dist = math.hypot(dx, dy)
        speed = _speed(s)
        T = max(1, math.ceil(dist / speed))
        pos = _predict_position(target.id, (target.x, target.y),
                                 initial_by_id, angular_velocity, step, step + T,
                                 comet_groups)
        if pos is None:
            return None
        arrival_pos = pos
        if tgt_owner == -1:
            needed = tgt_ships + 1
        else:
            needed = tgt_ships + tgt_prod * T + 1
            if tgt_owner != player:
                needed += ENEMY_CAPTURE_MARGIN
        if needed > available:
            return None
        if s >= needed and last_T == T:
            return s, T, arrival_pos
        s = max(s, needed)
        last_T = T
    if s <= available:
        return s, last_T or 1, arrival_pos
    return None


def _fleet_arrival_at(fleet, my_planet, initial_by_id, angular_velocity, step,
                      comet_groups):
    """Estimate (arrival_turn, hits) for an enemy fleet against my_planet.

    Approximation: project fleet by its constant velocity; project planet by
    its rotation/comet path; for each future tick up to maxT, test distance.
    Returns (T_turns, True) if it will collide within maxT, else (None, False).
    """
    speed = _speed(fleet.ships)
    fx, fy, ang = fleet.x, fleet.y, fleet.angle
    cx, cy = math.cos(ang) * speed, math.sin(ang) * speed
    max_T = 60  # ~60 turns is plenty given board diagonal ~141 / min speed 1
    pid = my_planet.id
    for t in range(1, max_T + 1):
        nx = fx + cx * t
        ny = fy + cy * t
        if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
            return None, False
        # Predict planet position at step+t
        pos = _predict_position(pid, (my_planet.x, my_planet.y),
                                 initial_by_id, angular_velocity, step, step + t,
                                 comet_groups)
        if pos is None:
            return None, False
        if math.hypot(nx - pos[0], ny - pos[1]) <= my_planet.radius + 0.5:
            return t, True
    return None, False


def agent(obs):
    if isinstance(obs, dict):
        getf = obs.get
    else:
        getf = lambda k, default=None: getattr(obs, k, default)

    player = getf("player", 0)
    raw_planets = getf("planets", []) or []
    raw_fleets = getf("fleets", []) or []
    initial_planets = getf("initial_planets", []) or []
    angular_velocity = getf("angular_velocity", 0.0) or 0.0
    step = getf("step", 0) or 0
    comet_groups = getf("comets", []) or []

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    if not my_planets:
        return []
    targets = [p for p in planets if p.owner != player]
    initial_by_id = {p[0]: p for p in initial_planets}
    game_remaining = max(1, EPISODE_STEPS - step)

    # -- 1. Defensive analysis: for each of my planets, sum incoming enemy --
    incoming = {p.id: [] for p in my_planets}
    for fleet in fleets:
        if fleet.owner == player:
            continue
        for mp in my_planets:
            t, hit = _fleet_arrival_at(fleet, mp, initial_by_id,
                                        angular_velocity, step, comet_groups)
            if hit:
                incoming[mp.id].append((t, fleet.ships))
                break  # one planet hit per fleet — first hit wins

    # For each planet, compute deficit = (enemy ships at earliest arrival) - garrison_at_arrival
    deficits = {}
    for mp in my_planets:
        threats = sorted(incoming[mp.id])
        if not threats:
            continue
        # Cumulative enemy ships through time t_i, vs garrison + my prod * t_i
        cum = 0
        worst_deficit = 0
        worst_t = None
        for t, eships in threats:
            cum += eships
            garrison = mp.ships + mp.production * t
            deficit = cum - garrison
            if deficit > worst_deficit:
                worst_deficit = deficit
                worst_t = t
        if worst_deficit > 0 and worst_t is not None:
            deficits[mp.id] = (worst_deficit, worst_t)

    # -- 2. Build attack candidates --
    attack_candidates = []
    for mine in my_planets:
        if mine.ships <= 1:
            continue
        for tgt in targets:
            est = _capture_cost(mine.x, mine.y, tgt,
                                 (tgt.x, tgt.y), mine.ships, step,
                                 initial_by_id, angular_velocity,
                                 comet_groups, player)
            if est is None:
                continue
            ships, T, arrival = est
            angle = _safe_launch_angle((mine.x, mine.y), arrival, tgt.radius)
            if angle is None:
                continue
            time_owned = max(0, game_remaining - T)
            value = tgt.production * time_owned
            if tgt.owner != -1:
                value += tgt.production * time_owned
            # Comets: extra incentive — they're free production with low cost
            if any(tgt.id in g["planet_ids"] for g in comet_groups):
                value *= 1.5
            score = value / (ships + 0.5 * T)
            attack_candidates.append((score, mine.id, tgt.id, ships, angle, T))

    attack_candidates.sort(reverse=True, key=lambda c: c[0])

    # -- 3. Build defense candidates: reinforcement fleets --
    # From each of my planets (not the threatened one) → threatened planet,
    # arriving by deadline. Score is "deficit covered per ship spent".
    defense_candidates = []
    my_by_id = {p.id: p for p in my_planets}
    for tid, (deficit, deadline) in deficits.items():
        target_mp = my_by_id[tid]
        for source in my_planets:
            if source.id == tid:
                continue
            # Try sending exactly the deficit; iterate for speed-vs-T fixed point
            s = max(1, deficit)
            last_T = None
            for _ in range(5):
                dist = math.hypot(target_mp.x - source.x,
                                  target_mp.y - source.y)
                speed = _speed(s)
                T = max(1, math.ceil(dist / speed))
                if T > deadline:
                    s = None
                    break
                # For reinforcement we don't need to predict planet movement
                # heavily; my own planet's position at arrival is roughly the
                # same orbital ring — close enough for sun-safety check.
                if s >= deficit and last_T == T:
                    break
                last_T = T
            if s is None or s > source.ships:
                continue
            arrival = _predict_position(tid, (target_mp.x, target_mp.y),
                                         initial_by_id, angular_velocity,
                                         step, step + last_T, comet_groups)
            if arrival is None:
                continue
            angle = _safe_launch_angle((source.x, source.y), arrival,
                                       target_mp.radius)
            if angle is None:
                continue
            # Score: covering the deficit is critical — high priority
            score = deficit / (s + 0.5 * last_T)
            defense_candidates.append((score, source.id, tid, s, angle,
                                       last_T, deficit))

    defense_candidates.sort(reverse=True, key=lambda c: c[0])

    # -- 4. Allocate ships: defense first (per-threatened-planet), then attack --
    used = {p.id: 0 for p in my_planets}
    available = {p.id: p.ships for p in my_planets}
    moves = []

    covered_deficit = {tid: 0 for tid in deficits}
    for score, sid, tid, ships, angle, T, deficit in defense_candidates:
        if covered_deficit[tid] >= deficit:
            continue
        if available[sid] - used[sid] < ships:
            continue
        used[sid] += ships
        covered_deficit[tid] += ships
        moves.append([sid, angle, int(ships)])

    targeted = set()
    for score, mid, tid, ships, angle, T in attack_candidates:
        if tid in targeted:
            continue
        if available[mid] - used[mid] < ships:
            continue
        used[mid] += ships
        targeted.add(tid)
        moves.append([mid, angle, int(ships)])

    # -- 5. Stacked neutral capture: if a neutral target wasn't taken and we
    # still have spare mines that individually fall short, pool them. --
    neutral_unclaimed = [t for t in targets
                          if t.owner == -1 and t.id not in targeted]
    for tgt in neutral_unclaimed:
        # Find willing contributors: mines with leftover ships
        contributors = []
        for mp in my_planets:
            spare = available[mp.id] - used[mp.id]
            if spare <= 1:
                continue
            # Use up to half spare, capped at target.ships+1
            send = min(spare - 1, tgt.ships + 2)
            if send <= 0:
                continue
            dist = math.hypot(tgt.x - mp.x, tgt.y - mp.y)
            T = max(1, math.ceil(dist / _speed(send)))
            arrival = _predict_position(tgt.id, (tgt.x, tgt.y), initial_by_id,
                                         angular_velocity, step, step + T,
                                         comet_groups)
            if arrival is None:
                continue
            angle = _safe_launch_angle((mp.x, mp.y), arrival, tgt.radius)
            if angle is None:
                continue
            contributors.append((mp.id, send, angle, T, dist))
        if not contributors:
            continue
        # Sort by distance ascending so first arrivals weaken the garrison
        contributors.sort(key=lambda c: c[4])
        total = sum(c[1] for c in contributors)
        if total <= tgt.ships:
            continue  # not enough even pooled; skip
        # Allocate from closest until target.ships+1 covered (a bit more for safety)
        budget = tgt.ships + 2
        for mid, send, angle, T, dist in contributors:
            if budget <= 0:
                break
            actual = min(send, budget)
            if actual <= 0:
                continue
            used[mid] += actual
            budget -= actual
            moves.append([mid, angle, int(actual)])
        targeted.add(tgt.id)

    return moves
