"""
Orbit Wars - Agent v3

v2d (conservation, dispatch math) + forward-simulation move filtering.

For each candidate move from v2d's heuristic, simulate 10 turns ahead with a
starter-like opponent model. Accept the move only if it improves the eval
(my_ships - opp_ships, plus production lookahead) over not taking it.

This filters out attacks whose targets get retaken or whose ships would have
been better off held in reserve.
"""

import math
from copy import deepcopy

# kaggle_environments.agent.get_last_callable appends the loaded file's
# directory to sys.path before exec'ing it, so sibling modules are
# importable directly.
import sim  # noqa: E402

# Shot-success classifier — predicts P(this fleet captures target).
# Falls back to no filter when model file is missing.
_SHOT_TREES = None
_SHOT_INIT = 0.0
try:
    import shot_model
    _SHOT_TREES = getattr(shot_model, "SHOT_TREES", None)
    _SHOT_INIT = getattr(shot_model, "SHOT_INIT", 0.0)
except Exception:
    pass

# V(state) → P(I win) classifier, trained on 4P self-play.
# Used to modulate shot_threshold dynamically:
#   high V → tighten (be conservative, protect lead)
#   low V → loosen (take risks, push for comeback)
_V_TREES = None
_V_BASE = 0.0
try:
    import value_4p_model as _vm
    _V_TREES = getattr(_vm, "V_TREES", None)
    _V_BASE = getattr(_vm, "V_BASE", 0.0)
except Exception:
    pass

SHOT_REJECT_THRESHOLD = 0.40   # default (fast-rot tight)
SHOT_THRESHOLD_SLOW = 0.20     # slow-rot loose

# 4P-specific overrides. Tightened relative to 2P because 3 opponents punish
# bad shots — but not too tight. v48: relaxed 4P (0.30→0.25 slow, 0.50→0.45
# fast) after bench showed +8pp 4P WR (12→20%) with 2P unchanged at 44%.
SHOT_THRESHOLD_4P_SLOW = 0.25
SHOT_THRESHOLD_4P_FAST = 0.45


def _v_feature_vector(planets, fleets, player, num_players, step):
    """16-d feature vector matching train_value_4p.py FEATURE_NAMES."""
    owner_planets = [0, 0, 0, 0]
    owner_ships = [0, 0, 0, 0]
    owner_prod = [0, 0, 0, 0]
    neutral_planets = 0
    my_planet_xy = []
    for p in planets:
        if p.owner == -1:
            neutral_planets += 1
        elif 0 <= p.owner < 4:
            owner_planets[p.owner] += 1
            owner_ships[p.owner] += p.ships
            owner_prod[p.owner] += p.production
            if p.owner == player:
                my_planet_xy.append((p.x, p.y))
    owner_fleet_ships = [0, 0, 0, 0]
    for f in fleets:
        if 0 <= f.owner < 4:
            owner_fleet_ships[f.owner] += f.ships

    my_ships = owner_ships[player] + owner_fleet_ships[player]
    my_planets_n = owner_planets[player]
    my_prod = owner_prod[player]
    my_fleet_ships = owner_fleet_ships[player]
    if my_planet_xy:
        cx = sum(x for x, _ in my_planet_xy) / len(my_planet_xy)
        cy = sum(y for _, y in my_planet_xy) / len(my_planet_xy)
        my_centrality = math.hypot(cx - 50, cy - 50)
    else:
        my_centrality = 0

    opp_indices = [i for i in range(4) if i != player]
    opps_ships = [owner_ships[i] + owner_fleet_ships[i] for i in opp_indices]
    opps_planets = [owner_planets[i] for i in opp_indices]
    opps_prod = [owner_prod[i] for i in opp_indices]
    opp_total_ships = sum(opps_ships)
    opp_total_planets = sum(opps_planets)
    opp_total_prod = sum(opps_prod)
    opp_fleet_ships = sum(owner_fleet_ships[i] for i in opp_indices)
    max_opp_ships = max(opps_ships, default=0)
    max_opp_planets = max(opps_planets, default=0)
    max_opp_prod = max(opps_prod, default=0)
    opp_count_alive = sum(1 for i in opp_indices
                          if owner_planets[i] > 0 or owner_ships[i] > 0)

    # Min distance from my planets to other-owner planets
    min_enemy_dist = 100.0
    for mp in my_planet_xy:
        for p in planets:
            if p.owner != player and p.owner != -1:
                d = math.hypot(mp[0] - p.x, mp[1] - p.y)
                if d < min_enemy_dist:
                    min_enemy_dist = d

    total_ships = max(1, my_ships + opp_total_ships)
    total_planets = max(1, my_planets_n + opp_total_planets + neutral_planets)
    total_prod = max(1, my_prod + opp_total_prod)
    total_fleet_ships = max(1, my_fleet_ships + opp_fleet_ships)

    def sd(a, b):
        return a / b if b > 1e-9 else 0.0

    return [
        step / 500.0,
        1.0 if num_players == 2 else 0.0,
        1.0 if num_players == 4 else 0.0,
        sd(my_ships, total_ships),
        sd(my_planets_n, total_planets),
        sd(my_prod, total_prod),
        my_centrality / 60.0,
        sd(max_opp_ships, total_ships),
        sd(max_opp_planets, total_planets),
        sd(max_opp_prod, total_prod),
        opp_count_alive / 3.0,
        min(1.0, min_enemy_dist / 100.0),
        sd(my_ships - opp_total_ships, total_ships),
        sd(my_planets_n - opp_total_planets, total_planets),
        sd(my_prod - opp_total_prod, total_prod),
        sd(my_fleet_ships, total_fleet_ships),
    ]

from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

MAX_SPEED = 6.0
SUN_BUFFER = 0.6
EPISODE_STEPS = 500
SIM_HORIZON = 16     # turns to look ahead
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


# -- Opponent model used inside sim rollouts: starter-like --

def _opp_starter_moves(state, opp_player):
    """Approximate starter's decision: each mine with >=20 ships sends half
    toward the closest static unowned target."""
    moves = []
    for p in state["planets"]:
        if p[1] != opp_player or p[5] < 16:
            continue
        x0, y0 = p[2], p[3]
        best = None; best_d = float('inf')
        for q in state["planets"]:
            if q[1] == opp_player:
                continue
            orbital_r = math.hypot(q[2] - CENTER, q[3] - CENTER)
            if orbital_r + q[4] < ROTATION_RADIUS_LIMIT:
                continue
            d = math.hypot(q[2] - x0, q[3] - y0)
            if d < best_d:
                best_d = d; best = q
        if best is None:
            continue
        ships = p[5] // 2
        if ships < 20:
            continue
        angle = math.atan2(best[3] - y0, best[2] - x0)
        moves.append([p[0], angle, ships])
    return moves


def _simulate(state, my_actions, opp_actions, horizon, player, num_players=2,
                opp_model="starter"):
    """Apply this turn's actions, then run `horizon` turns. opp_model decides
    what we expect the opponent to do each future turn:
      - "starter": each non-self player sends half-ship fleets at closest static
      - "none":    no new opp launches (only existing fleets resolve)
    """
    actions = [None] * num_players
    actions[player] = my_actions
    if opp_actions and num_players == 2:
        actions[1 - player] = opp_actions
    sim.step(state, actions)
    for _ in range(horizon - 1):
        actions = [None] * num_players
        actions[player] = []
        if opp_model == "starter":
            for opp in range(num_players):
                if opp == player:
                    continue
                opp_moves = _opp_starter_moves(state, opp)
                if opp_moves:
                    actions[opp] = opp_moves
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

    # FIX (4P bug): initial_planets at step 0 are still all-neutral (snapshot
    # taken before home assignment), so initial_owners was always empty and
    # num_players defaulted to 2. As a result, in 4P games main:
    #   - crashed at slots 2/3 with `actions[player]` IndexError → 0 launches
    #   - used 2P shot thresholds and 2P V-feature flags at slots 0/1
    # Compute num_players from current planets + fleets + self (more robust).
    all_owners = {p[1] for p in raw_planets if p[1] >= 0}
    for f in raw_fleets:
        if f[1] >= 0:
            all_owners.add(f[1])
    all_owners.add(player)
    max_player = max(all_owners) if all_owners else player
    # Env only supports 2 or 4 players; round up
    num_players = 4 if max_player >= 2 else 2

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
            candidates.append((base_value / (min_ships + 1.0 * T),
                                mine.id, tgt.id, min_ships, angle, T))
            # Variant 2: buffered capture. v49: in 4P, much bigger buffer
            # (v46 style — top Kaggle 4P used avg 24 ships); in 2P, keep v43
            # buffer (which works at 44% WR).
            if num_players == 4:
                buf = min(min_ships, max(8, int(avail * 0.6) - min_ships))
                discount = 0.99
            else:
                buf = min(int(min_ships * 0.5) + 5, avail - min_ships)
                discount = 0.95
            if buf > 0:
                big_ships = min_ships + buf
                candidates.append((base_value / (big_ships + 1.0 * T) * discount,
                                    mine.id, tgt.id, big_ships, angle, T))

    # === Map + format-conditional shot threshold ===
    is_4p = num_players == 4
    if is_4p:
        # 4P needs tighter filtering — 3 opponents punish bad shots
        shot_threshold = (SHOT_THRESHOLD_4P_SLOW if angular_velocity < 0.035
                          else SHOT_THRESHOLD_4P_FAST)
    else:
        shot_threshold = (SHOT_THRESHOLD_SLOW if angular_velocity < 0.035
                          else SHOT_REJECT_THRESHOLD)

    # === V-based threshold modulation (only if model loaded) ===
    # V predicts P(I eventually win) from current state.
    # If V > 0.7 → I'm winning → tighten threshold (don't risk).
    # If V < 0.3 → I'm losing → loosen threshold (be aggressive).
    if _V_TREES is not None:
        v_feats = _v_feature_vector(planets, fleets, player, num_players,
                                     step_now)
        z = _V_BASE
        for feat, thr, val, left, right in _V_TREES:
            node = 0
            while feat[node] >= 0:
                if v_feats[feat[node]] < thr[node]:
                    node = left[node]
                else:
                    node = right[node]
            z += val[node]
        try:
            p_win = 1.0 / (1.0 + math.exp(-z))
        except OverflowError:
            p_win = 1.0 if z > 0 else 0.0
        if p_win > 0.7:
            shot_threshold = min(0.65, shot_threshold + 0.10)
        elif p_win < 0.3:
            shot_threshold = max(0.10, shot_threshold - 0.15)

    # === Shot-reject filter (from learned classifier) ===
    if _SHOT_TREES is not None:
        my_planet_count = len(my_planets)
        my_total_ships = sum(p.ships for p in my_planets)
        opp_total_ships = sum(p.ships for p in planets
                              if p.owner != player and p.owner != -1)
        my_by_id = {p.id: p for p in my_planets}
        tgt_by_id = {p.id: p for p in planets}
        filtered = []
        for c in candidates:
            score, mid, tid, ships, angle, T = c
            src = my_by_id.get(mid)
            tgt = tgt_by_id.get(tid)
            if src is None or tgt is None:
                continue
            dist = math.hypot(tgt.x - src.x, tgt.y - src.y)
            speed = _speed(ships)
            eta = max(1, math.ceil(dist / speed))
            if tgt.owner == -1:
                owner_type = 0
                predicted_garrison = tgt.ships
            elif tgt.owner == player:
                owner_type = 2
                predicted_garrison = tgt.ships + tgt.production * eta
            else:
                owner_type = 1
                predicted_garrison = tgt.ships + tgt.production * eta
            margin = ships - predicted_garrison
            feats = [
                src.ships, src.production, tgt.ships, tgt.production,
                owner_type, dist, ships, speed, eta, predicted_garrison,
                margin, my_planet_count, my_total_ships, opp_total_ships,
                step_now / 500.0, num_players / 4.0,
                1.0 if num_players == 2 else 0.0,
            ]
            z = _SHOT_INIT
            for feat_list, thr_list, val_list, left_list, right_list in _SHOT_TREES:
                node = 0
                while feat_list[node] >= 0:
                    if feats[feat_list[node]] <= thr_list[node]:
                        node = left_list[node]
                    else:
                        node = right_list[node]
                z += val_list[node]
            # sigmoid
            try:
                p_success = 1.0 / (1.0 + math.exp(-z))
            except OverflowError:
                p_success = 1.0 if z > 0 else 0.0
            if p_success >= shot_threshold:
                filtered.append(c)
        candidates = filtered

    candidates.sort(reverse=True, key=lambda c: c[0])

    # Greedy sim-validated allocation
    sim_state_base = sim.make_state_from_obs(obs)
    baseline_state = sim.clone(sim_state_base)
    baseline_eval = _simulate(baseline_state, [], [], SIM_HORIZON, player,
                                num_players=num_players, opp_model="starter")

    accepted = []
    used = {p.id: 0 for p in my_planets}
    targeted = set()
    current_best = baseline_eval

    # Group candidates by (mid, tid). For each (mid, tid), present both
    # variants (min and buffered) and let sim pick whichever — or neither.
    K = min(15, len(candidates))
    grouped = {}
    for c in candidates[:K]:
        key = (c[1], c[2])  # (mid, tid)
        grouped.setdefault(key, []).append(c)

    # Visit groups in score order of their best variant
    group_keys = sorted(grouped.keys(),
                         key=lambda k: max(c[0] for c in grouped[k]),
                         reverse=True)
    for key in group_keys:
        mid, tid = key
        if tid in targeted:
            continue
        best_variant = None
        best_eval_for_group = current_best
        for score, _mid, _tid, ships, angle, T in grouped[key]:
            if max_launch[mid] - used[mid] < ships:
                continue
            trial_moves = accepted + [[mid, angle, int(ships)]]
            trial_state = sim.clone(sim_state_base)
            new_eval = _simulate(trial_state, trial_moves, [], SIM_HORIZON,
                                  player, num_players=num_players,
                                  opp_model="starter")
            if new_eval > best_eval_for_group:
                best_eval_for_group = new_eval
                best_variant = (ships, angle, T)
        if best_variant is not None:
            ships, angle, T = best_variant
            accepted = accepted + [[mid, angle, int(ships)]]
            current_best = best_eval_for_group
            used[mid] += ships
            targeted.add(tid)

    # Second pass: try dropping each accepted move; keep drops that improve.
    i = 0
    while i < len(accepted):
        trial_moves = accepted[:i] + accepted[i+1:]
        trial_state = sim.clone(sim_state_base)
        new_eval = _simulate(trial_state, trial_moves, [], SIM_HORIZON, player,
                              num_players=num_players, opp_model="starter")
        if new_eval > current_best:
            accepted = trial_moves
            current_best = new_eval
            # Don't increment i — the next move slid down into this slot
        else:
            i += 1

    return accepted
