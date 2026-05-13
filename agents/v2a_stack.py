"""
Orbit Wars - Agent v2a (stack-only)

v1 + comet bonus + neutral target stacking. No defense logic.
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6
EPISODE_STEPS = 500
ENEMY_CAPTURE_MARGIN = 2


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


def agent(obs):
    if isinstance(obs, dict):
        getf = obs.get
    else:
        getf = lambda k, default=None: getattr(obs, k, default)

    player = getf("player", 0)
    raw_planets = getf("planets", []) or []
    initial_planets = getf("initial_planets", []) or []
    angular_velocity = getf("angular_velocity", 0.0) or 0.0
    step = getf("step", 0) or 0
    comet_groups = getf("comets", []) or []

    planets = [Planet(*p) for p in raw_planets]
    my_planets = [p for p in planets if p.owner == player]
    if not my_planets:
        return []
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return []
    initial_by_id = {p[0]: p for p in initial_planets}
    game_remaining = max(1, EPISODE_STEPS - step)

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
            if any(tgt.id in g["planet_ids"] for g in comet_groups):
                value *= 1.5
            score = value / (ships + 0.5 * T)
            attack_candidates.append((score, mine.id, tgt.id, ships, angle, T))

    attack_candidates.sort(reverse=True, key=lambda c: c[0])

    used = {p.id: 0 for p in my_planets}
    available = {p.id: p.ships for p in my_planets}
    targeted = set()
    moves = []
    for score, mid, tid, ships, angle, T in attack_candidates:
        if tid in targeted:
            continue
        if available[mid] - used[mid] < ships:
            continue
        used[mid] += ships
        targeted.add(tid)
        moves.append([mid, angle, int(ships)])

    # Stacked neutral capture: pool spare ships across mines to take a neutral
    # too big for any single mine alone.
    neutral_unclaimed = [t for t in targets if t.owner == -1 and t.id not in targeted]
    for tgt in neutral_unclaimed:
        contributors = []
        for mp in my_planets:
            spare = available[mp.id] - used[mp.id]
            if spare <= 1:
                continue
            send_cap = min(spare - 1, tgt.ships + 2)
            if send_cap <= 0:
                continue
            dist = math.hypot(tgt.x - mp.x, tgt.y - mp.y)
            T = max(1, math.ceil(dist / _speed(send_cap)))
            arrival = _predict_position(tgt.id, (tgt.x, tgt.y), initial_by_id,
                                         angular_velocity, step, step + T,
                                         comet_groups)
            if arrival is None:
                continue
            angle = _safe_launch_angle((mp.x, mp.y), arrival, tgt.radius)
            if angle is None:
                continue
            contributors.append((mp.id, send_cap, angle, T, dist))
        if not contributors:
            continue
        contributors.sort(key=lambda c: c[4])
        if sum(c[1] for c in contributors) <= tgt.ships:
            continue
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
