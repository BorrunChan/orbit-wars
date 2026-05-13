"""
Orbit Wars - Agent v2c

v1 + careful defensive reinforcement:
  - Predict each enemy fleet's collision with our planets (continuous swept check).
  - For each threatened planet, only defend if it would actually fall
    (sequential combat math, accounting for production between arrivals).
  - Reinforce from the single closest mine that has enough spare ships
    and can arrive in time.
  - Bound defense spending per mine.

Otherwise same as v1.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6
EPISODE_STEPS = 500
MAX_DEFENSE_FRACTION = 0.6  # don't drain more than 60% of a mine on defense
DEFENSE_HORIZON = 40        # ticks to look ahead for incoming threats


def _speed(ships):
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
    sx, sy = src; tx, ty = dst
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
    chosen = sun_angle + (sun_half + 1e-3 if diff >= 0 else -sun_half - 1e-3)
    delta = (chosen - tgt_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(delta) > tgt_half - 1e-3:
        return None
    return chosen


def _capture_cost(mine_x, mine_y, target, predicted_dst, available, step,
                   initial_by_id, angular_velocity, comet_groups, player):
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
        if needed > available:
            return None
        if s >= needed and last_T == T:
            return s, T, arrival_pos
        s = max(s, needed)
        last_T = T
    if s <= available:
        return s, last_T or 1, arrival_pos
    return None


def _fleet_eta(fleet, my_planet, initial_by_id, angular_velocity, step,
                comet_groups):
    """Returns the turn offset at which the enemy fleet would collide with
    my_planet, or None. Discrete-tick approximation matching env stepping."""
    speed = _speed(fleet.ships)
    cx, cy = math.cos(fleet.angle) * speed, math.sin(fleet.angle) * speed
    for t in range(1, DEFENSE_HORIZON + 1):
        nx = fleet.x + cx * t
        ny = fleet.y + cy * t
        if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
            return None
        pos = _predict_position(my_planet.id, (my_planet.x, my_planet.y),
                                 initial_by_id, angular_velocity, step,
                                 step + t, comet_groups)
        if pos is None:
            return None
        if math.hypot(nx - pos[0], ny - pos[1]) <= my_planet.radius:
            return t
    return None


def _sequential_garrison(initial_ships, production, threats):
    """Simulate sequential same-turn-grouped combat. threats is a list of
    (t, ships) for incoming ENEMY fleets sorted by t.

    Returns the smallest (t, deficit) where deficit > 0 means the planet
    would fall at that arrival. Production accrues only while we still own
    the planet; if planet flips, no further owner-side production accrues.
    """
    # Group by t (same-turn fleets sum together).
    grouped = {}
    for t, sh in threats:
        grouped.setdefault(t, 0)
        grouped[t] += sh
    ts = sorted(grouped)
    garrison = initial_ships
    prev_t = 0
    for t in ts:
        garrison += production * (t - prev_t)
        e = grouped[t]
        if e > garrison:
            return t, e - garrison
        garrison -= e
        prev_t = t
    return None, 0


def _reinforce_eta(mine, target, ships, initial_by_id, angular_velocity,
                    step, comet_groups):
    """Travel time for a fleet of given size from mine to target."""
    last_T = None
    arrival = (target.x, target.y)
    for _ in range(5):
        dist = math.hypot(arrival[0] - mine.x, arrival[1] - mine.y)
        speed = _speed(ships)
        T = max(1, math.ceil(dist / speed))
        pos = _predict_position(target.id, (target.x, target.y),
                                 initial_by_id, angular_velocity, step,
                                 step + T, comet_groups)
        if pos is None:
            return None, None
        arrival = pos
        if last_T == T:
            return T, arrival
        last_T = T
    return last_T, arrival


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

    # ---- 1. Threat analysis ----
    threats_per_planet = {p.id: [] for p in my_planets}
    for fleet in fleets:
        if fleet.owner == player:
            continue
        # Find earliest collision among my planets
        best = None
        for mp in my_planets:
            t = _fleet_eta(fleet, mp, initial_by_id, angular_velocity, step,
                            comet_groups)
            if t is not None and (best is None or t < best[0]):
                best = (t, mp.id)
        if best is not None:
            threats_per_planet[best[1]].append((best[0], fleet.ships))

    # ---- 2. Build attack candidates (same as v1) ----
    attack_candidates = []
    for mine in my_planets:
        if mine.ships <= 1:
            continue
        for tgt in targets:
            est = _capture_cost(mine.x, mine.y, tgt, (tgt.x, tgt.y), mine.ships,
                                 step, initial_by_id, angular_velocity,
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
            score = value / (ships + 0.5 * T)
            attack_candidates.append((score, mine.id, tgt.id, ships, angle, T))

    attack_candidates.sort(reverse=True, key=lambda c: c[0])

    # ---- 3. Build defense moves ----
    used = {p.id: 0 for p in my_planets}
    available = {p.id: p.ships for p in my_planets}
    my_by_id = {p.id: p for p in my_planets}
    moves = []

    # For each threatened planet, compute deficit and dispatch reinforcement.
    for mpid, threats in threats_per_planet.items():
        if not threats:
            continue
        mp = my_by_id[mpid]
        # Sequential combat to find first-falling deadline.
        deadline, deficit = _sequential_garrison(mp.ships, mp.production,
                                                  sorted(threats))
        if deficit <= 0:
            continue
        # Find single closest mine that can spare deficit+1 in time.
        best = None
        for source in my_planets:
            if source.id == mpid:
                continue
            spare = available[source.id] - used[source.id]
            cap = int(source.ships * MAX_DEFENSE_FRACTION)
            spare = min(spare, cap)
            need = deficit + 1
            if spare < need:
                continue
            T, arrival = _reinforce_eta(source, mp, need, initial_by_id,
                                          angular_velocity, step, comet_groups)
            if T is None or T > deadline:
                continue
            angle = _safe_launch_angle((source.x, source.y), arrival, mp.radius)
            if angle is None:
                continue
            dist = math.hypot(source.x - mp.x, source.y - mp.y)
            if best is None or dist < best[0]:
                best = (dist, source.id, need, angle, T)
        if best is None:
            continue
        _, sid, ships, angle, T = best
        used[sid] += ships
        moves.append([sid, angle, int(ships)])

    # ---- 4. Attack allocation ----
    targeted = set()
    for score, mid, tid, ships, angle, T in attack_candidates:
        if tid in targeted:
            continue
        if available[mid] - used[mid] < ships:
            continue
        used[mid] += ships
        targeted.add(tid)
        moves.append([mid, angle, int(ships)])

    return moves
