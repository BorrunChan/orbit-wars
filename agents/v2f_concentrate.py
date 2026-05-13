"""
Orbit Wars - Agent v2f

v2d + concentration: cap attacks per mine per turn so a heavy mine doesn't
spread across too many cheap captures (which leaves it vulnerable to one
big enemy strike).
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6
EPISODE_STEPS = 500
DEFENSE_HORIZON = 40


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


def _max_dispatch(mp, threats):
    """Max ships we can launch from this mine this turn without the planet
    falling to any incoming enemy fleet. If the planet is doomed even with
    zero dispatch, returns full ship count (so ships escape elsewhere
    instead of being donated to the attacker)."""
    if not threats:
        return mp.ships
    grouped = {}
    for t, sh in threats:
        grouped[t] = grouped.get(t, 0) + sh
    ts = sorted(grouped)
    # For each threat time t, all earlier threats have already drained
    # cumulative `cum`. Constraint: ships + prod*t - L - cum >= e[t]
    # => L <= ships + prod*t - cum - e[t]
    cum = 0
    min_L_cap = float('inf')
    for t in ts:
        cap = mp.ships + mp.production * t - cum - grouped[t]
        min_L_cap = min(min_L_cap, cap)
        cum += grouped[t]
    if min_L_cap < 0:
        return mp.ships  # planet doomed regardless
    return max(0, int(min_L_cap))


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
    if not targets:
        return []
    initial_by_id = {p[0]: p for p in initial_planets}
    game_remaining = max(1, EPISODE_STEPS - step)

    # ---- Threat detection ----
    threats_per_planet = {p.id: [] for p in my_planets}
    for fleet in fleets:
        if fleet.owner == player:
            continue
        best = None
        for mp in my_planets:
            t = _fleet_eta(fleet, mp, initial_by_id, angular_velocity, step,
                            comet_groups)
            if t is not None and (best is None or t < best[0]):
                best = (t, mp.id)
        if best is not None:
            threats_per_planet[best[1]].append((best[0], fleet.ships))

    # Max launchable per mine — how many ships we can launch without losing.
    max_launch = {}
    for mine in my_planets:
        thr = threats_per_planet[mine.id]
        max_launch[mine.id] = _max_dispatch(mine, sorted(thr))

    # ---- Attack candidates ----
    candidates = []
    for mine in my_planets:
        avail_for_attack = max_launch[mine.id]
        if avail_for_attack <= 1:
            continue
        for tgt in targets:
            est = _capture_cost(mine.x, mine.y, tgt, (tgt.x, tgt.y),
                                 avail_for_attack, step, initial_by_id,
                                 angular_velocity, comet_groups, player)
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
            candidates.append((score, mine.id, tgt.id, ships, angle, T))

    candidates.sort(reverse=True, key=lambda c: c[0])

    used = {p.id: 0 for p in my_planets}
    attack_count = {p.id: 0 for p in my_planets}
    available = {p.id: p.ships for p in my_planets}
    targeted = set()
    moves = []
    for score, mid, tid, ships, angle, T in candidates:
        if tid in targeted:
            continue
        # Cap concurrent attacks from one mine: 2 in early/mid, 3 if we have
        # very high ship count there. Prevents one mine spreading too thin.
        mine_ships = available[mid]
        attack_cap = 1 if mine_ships < 25 else (2 if mine_ships < 80 else 3)
        if attack_count[mid] >= attack_cap:
            continue
        budget = min(available[mid] - used[mid], max_launch[mid] - used[mid])
        if budget < ships:
            continue
        used[mid] += ships
        attack_count[mid] += 1
        targeted.add(tid)
        moves.append([mid, angle, int(ships)])

    return moves
