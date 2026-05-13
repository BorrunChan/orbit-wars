"""
Orbit Wars - Agent v4 (aggressive)

v3 found the right candidates but used sim too restrictively: the "do nothing"
baseline against a starter-like opp model is too easy to beat passively, so
real captures looked worse than idling. Result: we hoarded ships while the
opponent (e.g. Kohaku) launched 30+ fleets in the first 75 turns.

v4 flips the default:
  - Take v2d's heuristic plan as the starting move set (greedy, all top-K
    candidates).
  - Use sim only as a SUBTRACTIVE pruner: try removing each move; remove
    if doing so improves eval.
  - Opp model uses minimum-capture launches every turn (more realistic than
    starter's wait-for-40-ships), and works in both 2P and 4P (loops over
    all enemy players, not just `1 - player`).
"""

import math
from copy import deepcopy

# kaggle_environments.agent.get_last_callable appends the loaded file's
# directory to sys.path before exec'ing it, so a sibling `sim.py` is
# importable directly.
import sim  # noqa: E402

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6
EPISODE_STEPS = 500
SIM_HORIZON = 8     # turns to look ahead
DEFENSE_HORIZON = 40


def _speed(ships):
    s = max(1, ships)
    return min(MAX_SPEED, 1.0 + (MAX_SPEED - 1.0) * (math.log(s) / math.log(1000)) ** 1.5)


def _predict_position(pid, current_pos, initial_by_id, angular_velocity,
                       step_now, future_step, comet_groups):
    for group in comet_groups:
        if pid in group["planet_ids"]:
            i = group["planet_ids"].index(pid)
            new_idx = group["path_index"] + (future_step - step_now)
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


def _capture_cost(mine_x, mine_y, target, predicted_dst, available, step_now,
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
                                 initial_by_id, angular_velocity, step_now,
                                 step_now + T, comet_groups)
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


def _fleet_eta(fleet, my_planet, initial_by_id, angular_velocity, step_now,
                comet_groups):
    spd = _speed(fleet.ships)
    cx, cy = math.cos(fleet.angle) * spd, math.sin(fleet.angle) * spd
    for t in range(1, DEFENSE_HORIZON + 1):
        nx = fleet.x + cx * t
        ny = fleet.y + cy * t
        if not (0 <= nx <= BOARD_SIZE and 0 <= ny <= BOARD_SIZE):
            return None
        pos = _predict_position(my_planet.id, (my_planet.x, my_planet.y),
                                 initial_by_id, angular_velocity, step_now,
                                 step_now + t, comet_groups)
        if pos is None:
            return None
        if math.hypot(nx - pos[0], ny - pos[1]) <= my_planet.radius:
            return t
    return None


def _doom_deficit(mp, threats):
    """If planet falls even with zero dispatch, return (failing_turn, deficit).
    Else None."""
    if not threats:
        return None
    grouped = {}
    for t, sh in threats:
        grouped[t] = grouped.get(t, 0) + sh
    ts = sorted(grouped)
    garrison = mp.ships
    prev_t = 0
    for t in ts:
        garrison += mp.production * (t - prev_t)
        e = grouped[t]
        if e > garrison:
            return t, e - garrison + 1
        garrison -= e
        prev_t = t
    return None


def _max_dispatch(mp, threats):
    if not threats:
        return mp.ships
    grouped = {}
    for t, sh in threats:
        grouped[t] = grouped.get(t, 0) + sh
    ts = sorted(grouped)
    cum = 0
    min_cap = float('inf')
    for t in ts:
        cap = mp.ships + mp.production * t - cum - grouped[t]
        min_cap = min(min_cap, cap)
        cum += grouped[t]
    if min_cap < 0:
        return mp.ships  # planet doomed, send everything elsewhere
    return max(0, int(min_cap))


# -- Opponent model used inside sim rollouts --

def _opp_aggressive_moves(state, my_player, num_players):
    """For each NON-self player, each owned planet attacks the closest
    enemy/neutral planet with the minimum capture force. Approximates an
    aggressive v1-like opponent. Works for 2P and 4P games.
    """
    out = [None] * num_players
    for opp in range(num_players):
        if opp == my_player:
            continue
        moves = []
        for p in state["planets"]:
            if p[1] != opp or p[5] <= 1:
                continue
            # find closest non-opp planet
            x0, y0 = p[2], p[3]
            best = None; best_d = float('inf')
            for q in state["planets"]:
                if q[1] == opp:
                    continue
                d = math.hypot(q[2] - x0, q[3] - y0)
                if d < best_d:
                    best_d = d; best = q
            if best is None:
                continue
            # min capture force (rough)
            send = min(p[5], best[5] + 1)
            if send <= 0:
                continue
            angle = math.atan2(best[3] - y0, best[2] - x0)
            moves.append([p[0], angle, int(send)])
        if moves:
            out[opp] = moves
    return out


def _simulate(state, my_actions, horizon, player, num_players, opp="aggressive"):
    """Apply this turn's actions, then run `horizon` more turns with opp model.

    opp: "aggressive" — opp launches min-capture every turn from every planet
         "none"       — opp does nothing (only their existing fleets resolve)
    """
    actions = [None] * num_players
    actions[player] = my_actions
    if opp == "aggressive":
        opp_now = _opp_aggressive_moves(state, player, num_players)
        for i in range(num_players):
            if i != player and opp_now[i]:
                actions[i] = opp_now[i]
    sim.step(state, actions)
    for _ in range(horizon - 1):
        actions = [None] * num_players
        actions[player] = []
        if opp == "aggressive":
            opp_moves = _opp_aggressive_moves(state, player, num_players)
            for i in range(num_players):
                if i != player and opp_moves[i]:
                    actions[i] = opp_moves[i]
        sim.step(state, actions)
    return sim.evaluate(state, player)


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
    step_now = getf("step", 0) or 0
    comet_groups = getf("comets", []) or []

    # Detect number of players from initial home assignment.
    initial_owners = {p[1] for p in initial_planets if p[1] != -1}
    num_players = max(initial_owners) + 1 if initial_owners else 2
    num_players = max(2, num_players)

    planets = [Planet(*p) for p in raw_planets]
    fleets = [Fleet(*f) for f in raw_fleets]
    my_planets = [p for p in planets if p.owner == player]
    if not my_planets:
        return []
    targets = [p for p in planets if p.owner != player]
    if not targets:
        return []
    initial_by_id = {p[0]: p for p in initial_planets}
    game_remaining = max(1, EPISODE_STEPS - step_now)

    # Threats
    threats_per_planet = {p.id: [] for p in my_planets}
    for fleet in fleets:
        if fleet.owner == player:
            continue
        best = None
        for mp in my_planets:
            t = _fleet_eta(fleet, mp, initial_by_id, angular_velocity,
                            step_now, comet_groups)
            if t is not None and (best is None or t < best[0]):
                best = (t, mp.id)
        if best is not None:
            threats_per_planet[best[1]].append((best[0], fleet.ships))

    max_launch = {}
    for mine in my_planets:
        thr = threats_per_planet[mine.id]
        max_launch[mine.id] = _max_dispatch(mine, sorted(thr))

    # Build candidate attack list. For each (mine, target), also offer a
    # "buffered" variant that sends extra ships so the captured planet
    # survives a counter-strike. Sim decides which (if any) to take.
    candidates = []
    for mine in my_planets:
        avail = max_launch[mine.id]
        if avail <= 1:
            continue
        for tgt in targets:
            est = _capture_cost(mine.x, mine.y, tgt, (tgt.x, tgt.y), avail,
                                 step_now, initial_by_id, angular_velocity,
                                 comet_groups, player)
            if est is None:
                continue
            min_ships, T, arrival = est
            angle = _safe_launch_angle((mine.x, mine.y), arrival, tgt.radius)
            if angle is None:
                continue
            time_owned = max(0, game_remaining - T)
            base_value = tgt.production * time_owned
            if tgt.owner != -1:
                base_value += tgt.production * time_owned
            # Variant 1: minimum capture
            candidates.append((base_value / (min_ships + 0.5 * T),
                                mine.id, tgt.id, min_ships, angle, T))
            # Variant 2: buffered capture (~ +5 or +50% whichever is smaller)
            buf = min(int(min_ships * 0.5) + 5, avail - min_ships)
            if buf > 0:
                big_ships = min_ships + buf
                # Re-derive speed/T for new fleet size (faster)
                # Keep angle/arrival approximation (target moves slightly less)
                candidates.append((base_value / (big_ships + 0.5 * T) * 0.95,
                                    mine.id, tgt.id, big_ships, angle, T))

    candidates.sort(reverse=True, key=lambda c: c[0])

    # --- Phase 1: heuristic greedy plan (v2d-style, no sim) ---
    # For each (mid, tid), pick the BEST-SCORING variant (min or buffered),
    # respecting per-mine ship budget and one-target-per-attack constraints.
    heuristic = []
    used = {p.id: 0 for p in my_planets}
    targeted = set()
    K = min(60, len(candidates))
    for score, mid, tid, ships, angle, T in candidates[:K]:
        if tid in targeted:
            continue
        if max_launch[mid] - used[mid] < ships:
            continue
        heuristic.append([mid, angle, int(ships)])
        used[mid] += ships
        targeted.add(tid)

    return heuristic
