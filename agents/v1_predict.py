"""
Orbit Wars - Agent v1

Improvements over v0:
  - Predicts target position at arrival for rotating planets / comets
  - Accounts for production growth during fleet transit
  - Iteratively solves fleet-size <-> travel-time fixed point (bigger fleets fly faster)
  - Avoids the sun by deflecting launch angle to tangent of the sun's exclusion cone
  - Globally assigns mines to targets by score, avoiding double-targeting
  - Scores targets by future-production value, with a denial bonus for enemy planets
"""

import math
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6      # extra clearance beyond SUN_RADIUS when checking sun-blocked paths
EPISODE_STEPS = 500


def _speed(ships: int) -> float:
    s = max(1, ships)
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5)


def _predict_position(pid, current_pos, initial_by_id, angular_velocity,
                      step, future_step, comet_groups):
    """Return (x, y) of planet at future_step. None if a comet that has expired."""
    # Comet? Use its path.
    for group in comet_groups:
        if pid in group["planet_ids"]:
            i = group["planet_ids"].index(pid)
            new_idx = group["path_index"] + (future_step - step)
            path = group["paths"][i]
            if 0 <= new_idx < len(path):
                return (path[new_idx][0], path[new_idx][1])
            return None  # expired
    init = initial_by_id.get(pid)
    if init is None:
        return current_pos
    dx, dy = init[2] - CENTER, init[3] - CENTER
    r = math.hypot(dx, dy)
    # Rotation check uses the planet's own radius (4th index)
    if r + init[4] >= ROTATION_RADIUS_LIMIT:
        return (init[2], init[3])
    a0 = math.atan2(dy, dx)
    a1 = a0 + angular_velocity * future_step
    return (CENTER + r * math.cos(a1), CENTER + r * math.sin(a1))


def _safe_launch_angle(src, dst, target_radius):
    """Return an angle from src that (a) lands within target_radius of dst and
    (b) keeps the straight ray outside the sun. None if no such angle exists."""
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
        return tgt_angle  # source inside sun (shouldn't happen)
    sun_angle = math.atan2(sdy, sdx)
    sun_half = math.asin(min(1.0, R / sdist))

    if tdist <= target_radius:
        return tgt_angle
    tgt_half = math.asin(min(1.0, target_radius / tdist))

    diff = (tgt_angle - sun_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(diff) >= sun_half + tgt_half:
        return tgt_angle  # cones don't overlap, straight shot is clean

    # Conflict: deflect to side of target away from sun core
    if diff >= 0:
        chosen = sun_angle + sun_half + 1e-3
    else:
        chosen = sun_angle - sun_half - 1e-3
    delta = (chosen - tgt_angle + math.pi) % (2 * math.pi) - math.pi
    if abs(delta) > tgt_half - 1e-3:
        return None  # deflected angle misses the target body
    return chosen


def _capture_cost(mine_x, mine_y, target, predicted_dst, available, step,
                   initial_by_id, angular_velocity, comet_groups):
    """Iteratively solve for minimum fleet to capture target at arrival.

    target is a Planet namedtuple. Iterates a few times to converge on
    (ships, travel_turns, predicted_arrival_pos)."""
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
        # Re-predict at this T (movement happens once per turn; arrival_step = step+T)
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


def agent(obs):
    # Tolerate both dict-style and namespace-style observations.
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

    candidates = []
    for mine in my_planets:
        if mine.ships <= 1:
            continue
        for tgt in targets:
            est = _capture_cost(mine.x, mine.y, tgt,
                                 (tgt.x, tgt.y), mine.ships, step,
                                 initial_by_id, angular_velocity, comet_groups)
            if est is None:
                continue
            ships, T, arrival = est
            angle = _safe_launch_angle((mine.x, mine.y), arrival, tgt.radius)
            if angle is None:
                continue
            # Value: future production we'd accumulate after capture.
            time_owned = max(0, game_remaining - T)
            value = tgt.production * time_owned
            if tgt.owner != -1:
                # Denying enemy production is worth roughly the same.
                value += tgt.production * time_owned
            # Slight bias toward closer / cheaper captures.
            score = value / (ships + 0.5 * T)
            candidates.append((score, mine.id, tgt.id, ships, angle, T))

    candidates.sort(reverse=True, key=lambda c: c[0])

    used = {p.id: 0 for p in my_planets}
    available = {p.id: p.ships for p in my_planets}
    targeted = set()
    moves = []
    for score, mid, tid, ships, angle, T in candidates:
        if tid in targeted:
            continue
        if available[mid] - used[mid] < ships:
            continue
        used[mid] += ships
        targeted.add(tid)
        moves.append([mid, angle, int(ships)])

    return moves


# Expose under the canonical name for Kaggle submission.
def agent_function(obs):
    return agent(obs)
