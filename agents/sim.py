"""
Lightweight forward simulator for Orbit Wars.

State is a plain dict so we can deep-copy it cheaply between trial moves.
Implements the subset of env logic needed for short-horizon rollouts:
  - Fleet launch (validation: planet ownership, ship count)
  - Production
  - Fleet movement, sun/bounds/planet collision
  - Planet rotation, comet path advance
  - Combat resolution

Skipped (not needed for short horizons):
  - Comet spawning (we assume no new comets within sim window)
  - Comet expiration is best-effort
"""

import math
from copy import deepcopy

CENTER = 50.0
BOARD_SIZE = 100.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_SPEED = 6.0


def _speed(ships):
    s = max(1, ships)
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5)


def _seg_dist_to_origin(v, w):
    l2 = (v[0]-w[0])**2 + (v[1]-w[1])**2
    if l2 == 0:
        return math.hypot(CENTER - v[0], CENTER - v[1])
    t = max(0, min(1, ((CENTER - v[0])*(w[0]-v[0]) + (CENTER - v[1])*(w[1]-v[1])) / l2))
    px = v[0] + t*(w[0] - v[0])
    py = v[1] + t*(w[1] - v[1])
    return math.hypot(CENTER - px, CENTER - py)


def _swept_pair_hit(A, B, P0, P1, r):
    d0x, d0y = A[0] - P0[0], A[1] - P0[1]
    dvx = (B[0] - A[0]) - (P1[0] - P0[0])
    dvy = (B[1] - A[1]) - (P1[1] - P0[1])
    a = dvx * dvx + dvy * dvy
    b = 2.0 * (d0x * dvx + d0y * dvy)
    c = d0x * d0x + d0y * d0y - r * r
    if a < 1e-12:
        return c <= 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return t2 >= 0.0 and t1 <= 1.0


def make_state_from_obs(obs):
    """Convert agent obs into a self-contained sim state."""
    def g(k, default=None):
        if isinstance(obs, dict):
            return obs.get(k, default)
        return getattr(obs, k, default)
    state = {
        "step": g("step", 0) or 0,
        "planets": [list(p) for p in g("planets", []) or []],
        "fleets": [list(f) for f in g("fleets", []) or []],
        "initial_planets": [list(p) for p in g("initial_planets", []) or []],
        "angular_velocity": g("angular_velocity", 0.0) or 0.0,
        "comets": deepcopy(g("comets", []) or []),
        "comet_planet_ids": list(g("comet_planet_ids", []) or []),
        "next_fleet_id": (max([f[0] for f in g("fleets", []) or []] or [0]) + 1) or 0,
    }
    return state


def clone(state):
    return {
        "step": state["step"],
        "planets": [p[:] for p in state["planets"]],
        "fleets": [f[:] for f in state["fleets"]],
        "initial_planets": state["initial_planets"],  # immutable
        "angular_velocity": state["angular_velocity"],
        "comets": deepcopy(state["comets"]),  # path_index changes
        "comet_planet_ids": state["comet_planet_ids"],  # rarely changes mid-sim
        "next_fleet_id": state["next_fleet_id"],
    }


def step(state, actions):
    """Advance one tick. `actions` is a list of [moves_per_player].

    moves_per_player is a list of [from_pid, angle, ships] for that player."""
    planets = state["planets"]
    fleets = state["fleets"]
    angular_velocity = state["angular_velocity"]
    step_num = state["step"]
    comet_pid_set = set(state["comet_planet_ids"])

    # Skip comet spawn/expiration in the inner sim (rare events).

    # 1. Fleet launch
    pid_to_planet = {p[0]: p for p in planets}
    for player_id, moves in enumerate(actions):
        if not moves:
            continue
        for move in moves:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            p = pid_to_planet.get(from_id)
            if p is None or p[1] != player_id or ships <= 0 or p[5] < ships:
                continue
            p[5] -= ships
            start_x = p[2] + math.cos(angle) * (p[4] + 0.1)
            start_y = p[3] + math.sin(angle) * (p[4] + 0.1)
            fleets.append([state["next_fleet_id"], player_id, start_x, start_y,
                           angle, from_id, ships])
            state["next_fleet_id"] += 1

    # 2. Production
    for p in planets:
        if p[1] != -1:
            p[5] += p[6]

    # 3. Compute end-of-tick planet positions
    initial_by_id = {p[0]: p for p in state["initial_planets"]}
    planet_paths = {}  # pid -> (old, new, check_collision)
    expired_comets = []

    for p in planets:
        pid = p[0]
        if pid in comet_pid_set:
            continue
        old = (p[2], p[3])
        new = old
        init = initial_by_id.get(pid)
        if init is not None:
            dx, dy = init[2] - CENTER, init[3] - CENTER
            r = math.hypot(dx, dy)
            if r + p[4] < ROTATION_RADIUS_LIMIT:
                a0 = math.atan2(dy, dx)
                a1 = a0 + angular_velocity * (step_num + 1)
                new = (CENTER + r * math.cos(a1), CENTER + r * math.sin(a1))
        planet_paths[pid] = (old, new, True)

    for group in state["comets"]:
        group["path_index"] += 1
        idx = group["path_index"]
        for i, pid in enumerate(group["planet_ids"]):
            p = pid_to_planet.get(pid)
            if p is None:
                continue
            old = (p[2], p[3])
            path = group["paths"][i]
            if idx >= len(path):
                expired_comets.append(pid)
                planet_paths[pid] = (old, old, True)
            else:
                new = (path[idx][0], path[idx][1])
                check = old[0] >= 0
                planet_paths[pid] = (old, new, check)

    # 4. Fleet movement + collisions
    fleets_to_remove = []
    combat_lists = {p[0]: [] for p in planets}

    for f in fleets:
        angle = f[4]
        ships = f[6]
        spd = _speed(ships)
        old_pos = (f[2], f[3])
        f[2] += math.cos(angle) * spd
        f[3] += math.sin(angle) * spd
        new_pos = (f[2], f[3])
        hit = False
        for p in planets:
            path = planet_paths.get(p[0])
            if path is None or not path[2]:
                continue
            p_old, p_new, _ = path
            if _swept_pair_hit(old_pos, new_pos, p_old, p_new, p[4]):
                combat_lists[p[0]].append(f)
                fleets_to_remove.append(f)
                hit = True
                break
        if hit:
            continue
        if not (0 <= f[2] <= BOARD_SIZE and 0 <= f[3] <= BOARD_SIZE):
            fleets_to_remove.append(f)
            continue
        if _seg_dist_to_origin(old_pos, new_pos) < SUN_RADIUS:
            fleets_to_remove.append(f)

    # 5. Apply planet motion
    for p in planets:
        path = planet_paths.get(p[0])
        if path is not None:
            p[2], p[3] = path[1]

    # 6. Remove expired comets
    if expired_comets:
        es = set(expired_comets)
        state["planets"] = [p for p in state["planets"] if p[0] not in es]
        state["comet_planet_ids"] = [pid for pid in state["comet_planet_ids"]
                                       if pid not in es]
        for g in state["comets"]:
            g["planet_ids"] = [pid for pid in g["planet_ids"] if pid not in es]
        state["comets"] = [g for g in state["comets"] if g["planet_ids"]]
        planets = state["planets"]

    state["fleets"] = [f for f in fleets if f not in fleets_to_remove]

    # 7. Combat
    pid_to_planet = {p[0]: p for p in planets}
    for pid, fleet_list in combat_lists.items():
        p = pid_to_planet.get(pid)
        if not p or not fleet_list:
            continue
        ship_by_owner = {}
        for f in fleet_list:
            ship_by_owner[f[1]] = ship_by_owner.get(f[1], 0) + f[6]
        if not ship_by_owner:
            continue
        sorted_p = sorted(ship_by_owner.items(), key=lambda x: x[1], reverse=True)
        top_p, top_s = sorted_p[0]
        if len(sorted_p) > 1:
            second_s = sorted_p[1][1]
            survivor_s = top_s - second_s
            if top_s == second_s:
                survivor_s = 0
            survivor_owner = top_p if survivor_s > 0 else -1
        else:
            survivor_owner = top_p
            survivor_s = top_s
        if survivor_s > 0:
            if p[1] == survivor_owner:
                p[5] += survivor_s
            else:
                p[5] -= survivor_s
                if p[5] < 0:
                    p[1] = survivor_owner
                    p[5] = abs(p[5])

    state["step"] = step_num + 1
    return state


def evaluate(state, player):
    """Score = (my total ships) - (opp total ships), with production weighting."""
    my_ships = opp_ships = 0
    my_prod = opp_prod = 0
    for p in state["planets"]:
        if p[1] == player:
            my_ships += p[5]; my_prod += p[6]
        elif p[1] != -1:
            opp_ships += p[5]; opp_prod += p[6]
    for f in state["fleets"]:
        if f[1] == player:
            my_ships += f[6]
        else:
            opp_ships += f[6]
    remaining = max(1, 500 - state["step"])
    return (my_ships - opp_ships) + (my_prod - opp_prod) * remaining * 0.3
