"""
v138 — v133 + 2 Producer-baseline concepts ported to pure Python.

Ported from Slawek Biel's "The Producer" (Kaggle ~1216, rank 74):
  A. Always-on pressure-gradient regroup. Replaces v130 logistics layer which
     was gated off during swarm — that gate was the root cause of v130's 778
     vs v98's 803.
  C. safe_drain: closed-form max ships a source can shed while staying held
     over horizon H, replacing v133's "ships − current attackers" heuristic.

Skipped from Producer's library:
  - capture_floor reinforcement margin (defined in their library but not
    enabled in their agent; v133 calculate_req_ships already accumulates
    production over ETA).
  - friendly_flip_targets (v133 reinforcement_plans + under_attack covers it).

Builds on v133 (v131 + 4P concentrated-strike fix). 2P concentrated strike,
MIN_SHIPS=8, neutral bonus, all retained.
"""
import math
import kaggle_environments.envs.orbit_wars.orbit_wars as ow

fleet_trajectories = []
reinforcement_trajectories = []
moving_planets = []
planets_coords = {}
steps = 0

MAX_SPEED = 6.0
# could use RL in future to tune these vars to optimal values
MIN_SHIPS_MINE_ATTACK = 8
MIN_SHIPS_TARGET_COOP_ATTACK = 20
COOP_PLANET_CAP = 8
COLLIDE_TICK_THOLD = 1

FORMULA_DIST = 100
FORMULA_PROD_MULT = 15
FORMULA_ENEMY_BONUS_MULT = 10
FORMULA_NEUTRAL_BONUS_MULT = 6
FORMULA_TOTAL_SHIPS_PERCENT = 0.7

# v138 Producer-baseline ports
REGROUP_HORIZON = 7              # max eta (turns) for a regroup hop
REGROUP_GAP_FRAC = 0.25          # gap > 25% of max pressure required to fire
REGROUP_MIN_SHIPS = 8
SAFE_DRAIN_HORIZON = 12          # H steps the source must remain held


def get_custom_score(m, t):
    dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)

    min_ships = t.ships + 1
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, min_ships)) / math.log(1000)) ** 1.5
    eta = dist / fleet_speed

    enemy_produced = 0
    enemy_bonus = 0
    neutral_bonus = 0
    if t.owner == -1:
        neutral_bonus = t.production
    else:
        enemy_produced = eta * t.production
        enemy_bonus = t.production

    total_ships = min_ships + enemy_produced

    return (
        (FORMULA_DIST - dist)
        + (FORMULA_PROD_MULT * t.production)
        + (FORMULA_ENEMY_BONUS_MULT * enemy_bonus)
        + (FORMULA_NEUTRAL_BONUS_MULT * neutral_bonus)
        - (FORMULA_TOTAL_SHIPS_PERCENT * total_ships)
        - (2 * eta)
    )

            

def get_planets_under_attack(mine, fleets, player, vel):
    mov_pl_traj = {}
    under_attack = {}
    seen = set()
    fleets = [f for f in fleets if f.owner != player]
    for m in mine:
        if m.id in moving_planets:
            mov_pl_traj[m.id] = get_planet_trajectories(m, vel)

    for f in fleets:
        fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(f.ships) / math.log(1000)) ** 1.5
        prev_x = f.x
        prev_y = f.y

        for tick in range(1, 61):
            next_x = f.x + math.cos(f.angle) * fleet_speed * tick
            next_y = f.y + math.sin(f.angle) * fleet_speed * tick

            for m in mine:
                if m.id in moving_planets:
                    m_x, m_y = mov_pl_traj[m.id][tick-1] # tick is 1 based, index 0 based, so -1
                else:
                    m_x, m_y = m.x, m.y
                    
                if collides(prev_x, prev_y, next_x, next_y, m_x, m_y, m.radius): 
                    if (m.id, f.id) not in seen:
                        if m.id not in under_attack:
                            under_attack[m.id] = {
                                "planet": m,
                                "fleets": []
                            }
                            
                        under_attack[m.id]["fleets"].append({
                            "fleet": f,
                            "arrive_tick": tick
                        })
                        seen.add((m.id, f.id))
            
            prev_x = next_x
            prev_y = next_y            
                    
    return under_attack
    


def refresh_local_obs(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    mine = [p for p in planets if p.owner == obs.get("player", [])]
    targets = [p for p in planets if p.owner != obs.get("player", [])]
    player = obs.get("player", -2)
    fleets = [ow.Fleet(*f) for f in obs.get("fleets", [])]

    return {
        "planets": planets,
        "mine": mine,
        "targets": targets,
        "player": player,
        "fleets": fleets
    }

def sun_collision(m, fleet_speed, angle, ticks=61):
    prev_x = m.x
    prev_y = m.y

    for tick in range(1, ticks):
        x = m.x + math.cos(angle) * fleet_speed * tick
        y = m.y + math.sin(angle) * fleet_speed * tick

        if collides(prev_x, prev_y, x, y, 50, 50, 10):
            return True

        prev_x = x
        prev_y = y
            
    return False


def calculate_req_ships_moving(attacking_planets, t, base_ships, vel):
    MAX_SPEED = 6.0
    required_ships = base_ships
    planet_trajectories = get_planet_trajectories(t, vel)
    
    for _ in range(3):
        remainder = required_ships
        max_tick = 0

        for a_p in attacking_planets:
            p = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)

            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))

            if p_ships <= 0:
                continue
            
            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, p_ships)) / math.log(1000)) ** 1.5

            found_tick = 0
            for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
                dist = math.sqrt((p.x - tx)**2 + (p.y - ty)**2)
                turns_to_arrive = math.floor(dist / fleet_speed)

                if abs(turns_to_arrive - tick) <= 1:
                    found_tick = tick
                    break

            if found_tick > max_tick:
                max_tick = found_tick

            remainder -= p_ships

        new_req = base_ships + (max_tick * t.production)
        if new_req == required_ships:
            break
        required_ships = new_req
        
    return required_ships

def calculate_req_ships(attacking_planets, t, base_ships):
    required_ships = base_ships
    
    for _ in range(3):
        remainder = required_ships
        max_tick = 0
        
        for a_p in attacking_planets:
            p = a_p["planet"]
            p_ships = min(a_p["ships"], remainder)
            
            if p_ships > 0:
                p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))

            if p_ships <= 0:
                continue
            
            ships_for_speed = max(1, p_ships)
            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships_for_speed) / math.log(1000)) ** 1.5
            dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
            tick_arrival = math.floor(dist / fleet_speed)
            
            if tick_arrival > max_tick:
                max_tick = tick_arrival

            remainder -= p_ships

        new_req = base_ships + (max_tick * t.production)
        
        if new_req == required_ships:
            break
            
        required_ships = new_req
    
    return required_ships


def calculate_angle(m, t):
    return math.atan2(t.y - m.y, t.x - m.x)
    

def find_angle_to_moving_planet(p, t, ships, vel):
    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(ships) / math.log(1000)) ** 1.5
    planet_trajectories = get_planet_trajectories(t, vel)

    for tick, (tx, ty) in enumerate(planet_trajectories, start=1):
        dx = tx - p.x
        dy = ty - p.y
        dist_to_target = math.sqrt(dx**2 + dy**2) - p.radius

        travel_dist = fleet_speed * tick
        miss_dist = abs(travel_dist - dist_to_target)

        if miss_dist > t.radius:
            continue
        
        angle = math.atan2(dy, dx)
        
        if sun_collision(p, fleet_speed, angle):
            return None, None

        return angle, tick

    return None, None


def collides(x1, y1, x2, y2, cx, cy, r):
    vec_x = x2 - x1
    vec_y = y2 - y1

    vec_to_cx = cx - x1
    vec_to_cy = cy - y1

    vec_length_sq = vec_x**2 + vec_y**2

    if vec_length_sq == 0:
        dx = x1 - cx
        dy = y1 - cy
        return dx**2 + dy**2 <= r**2

    closest_point = (vec_to_cx * vec_x + vec_to_cy * vec_y) / vec_length_sq
    closest_point = max(0, min(1, closest_point))

    closest_x = x1 + closest_point * vec_x
    closest_y = y1 + closest_point * vec_y

    dx = closest_x - cx
    dy = closest_y - cy
    return dx**2 + dy**2 <= r**2


def get_closest_planets_to_target(mine, t):
    planets = []
    for m in mine:
        dist = math.sqrt((m.x - t.x)**2 + (m.y - t.y)**2)
        planets.append((m, dist))
    planets = sorted(planets, key=lambda k: k[1])
    return planets
    

def update_fleet_trajectories(fleets):
    for f_t in fleet_trajectories[:]:
        found = False
        for f in fleets:
            if f.from_planet_id == f_t["mine"].id and abs(f.angle - f_t["angle"]) < 1e-3:
                found = True
                break

        if found:
            f_t["arrive_tick"] = max(0, f_t["arrive_tick"] - 1)

        if not found:
            fleet_trajectories.remove(f_t)


def update_reinforcement_trajectories(planets):
    planet_ids = {p.id for p in planets}
    
    for r_t in reinforcement_trajectories[:]:
        r_t["arrive_tick"] -= 1

        if r_t["arrive_tick"] <= 0:
            reinforcement_trajectories.remove(r_t)
            continue


def get_planet_trajectories(p, vel):
    planet_trajectories = []
    angle = math.atan2(p.y - 50, p.x - 50)
    r = math.sqrt((p.x - 50)**2 + (p.y - 50)**2)
    for tick in range(1, 61): # max 60 ticks
        angle_t = angle + vel * tick
        x_t = 50 + r * math.cos(angle_t)
        y_t = 50 + r * math.sin(angle_t)
        planet_trajectories.append((x_t, y_t))

    return planet_trajectories
    

def fill_moving_planets(obs):
    planets = [ow.Planet(*p) for p in obs.get("planets", [])]
    initial_by_id = {i[0]: ow.Planet(*i) for i in obs.get("initial_planets", [])}
    for p in planets:
        i = initial_by_id[p.id]
        if (p.x, p.y) != (i.x, i.y):
            if p.id not in moving_planets:
                moving_planets.append(p.id)

def get_reinforcement_plans(mine, under_attack):
    reinforcement_plans = {}
    
    for p in mine:
        if p.id in under_attack:
            attacking_fleets = sorted(
                under_attack[p.id]["fleets"],
                key=lambda att: att["arrive_tick"]
            )
            
            incoming_reinforcements = sorted(
                [r for r in reinforcement_trajectories if r["target"].id == p.id],
                key=lambda r: r["arrive_tick"]
            )
            
            p_available_ships = p.ships
            previous_tick = 0
            r_idx = 0

            for att in attacking_fleets:
                att_arrive_tick = att["arrive_tick"]

                p_available_ships += (att_arrive_tick - previous_tick) * p.production
                
                while (
                    r_idx < len(incoming_reinforcements)
                    and incoming_reinforcements[r_idx]["arrive_tick"] <= att_arrive_tick
                ):
                    p_available_ships += incoming_reinforcements[r_idx]["total_ships"]
                    r_idx += 1

                enemy_ships = att["fleet"].ships
                p_available_ships -= enemy_ships
                previous_tick = att_arrive_tick
                
                if p_available_ships < 0:
                    reinforcements_needed = max(MIN_SHIPS_MINE_ATTACK, abs(p_available_ships))
                    reinforcement_plans[p] = {
                        "ships_needed": reinforcements_needed,
                        "needed_by_tick": att_arrive_tick
                    }
                    break
                
    return reinforcement_plans


# ---------------------------------------------------------------------------
# v138 Producer-baseline helpers
# ---------------------------------------------------------------------------

def _planet_fleet_speed(ships):
    return 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5


def compute_enemy_pressure(mine, enemy_planets, horizon=SAFE_DRAIN_HORIZON):
    """Per-owned-planet distance-decayed enemy ship mass — regroup gradient.

    pressure[m] = Σ_e enemy_ships[e] · max(0, 1 − d(m,e) / (speed_e · H))

    Nearer/heavier enemies weight more. Pure arithmetic, no projection.
    """
    pressure = {m.id: 0.0 for m in mine}
    if not enemy_planets:
        return pressure
    for m in mine:
        s = 0.0
        for e in enemy_planets:
            if e.ships <= 0:
                continue
            d = math.sqrt((m.x - e.x) ** 2 + (m.y - e.y) ** 2)
            speed_e = _planet_fleet_speed(e.ships)
            reach = speed_e * horizon
            if reach <= 1e-6:
                continue
            decay = 1.0 - d / reach
            if decay <= 0:
                continue
            s += e.ships * decay
        pressure[m.id] = s
    return pressure


def safe_drain_for_source(m, under_attack, reinforcement_trajectories,
                          horizon=SAFE_DRAIN_HORIZON):
    """Closed-form max ships m can shed while staying held over `horizon`.

    Walk t = 1..H. ships_held(t) = ships_now + t·prod − cum_attacks(t) + cum_reinforce(t).
    If ships_held(t) < 0 at any t, m flips under do-nothing → drain capped at
    the previous step's held count. Otherwise safe_drain = min_t ships_held(t),
    capped by ships_now. (Equivalent of orbit_lite.planner_core.safe_drain.)
    """
    ships_now = m.ships
    prod = m.production
    atk_fleets = under_attack.get(m.id, {}).get("fleets", []) if under_attack.get(m.id) else []
    attacks = sorted(
        [(att["arrive_tick"], att["fleet"].ships) for att in atk_fleets],
        key=lambda x: x[0],
    )
    reinforces = sorted(
        [(r["arrive_tick"], r["total_ships"])
         for r in reinforcement_trajectories
         if r["target"].id == m.id and r["arrive_tick"] >= 1],
        key=lambda x: x[0],
    )
    min_held = ships_now
    cum_atk = 0
    cum_rein = 0
    ai = ri = 0
    for t in range(1, horizon + 1):
        while ai < len(attacks) and attacks[ai][0] <= t:
            cum_atk += attacks[ai][1]
            ai += 1
        while ri < len(reinforces) and reinforces[ri][0] <= t:
            cum_rein += reinforces[ri][1]
            ri += 1
        held = ships_now + t * prod - cum_atk + cum_rein
        if held < 0:
            # planet flips at t under do-nothing — anything we drain now is forfeit
            return 0
        if held < min_held:
            min_held = held
    return max(0, min(int(min_held), int(ships_now)))


def plan_regroup(mine, ships_left, pressure, under_attack, vel,
                 max_eta=REGROUP_HORIZON, min_send=REGROUP_MIN_SHIPS):
    """Shuttle leftover ships from low- to high-pressure mine planets.

    Always-on (no swarm gate). Respects gap > REGROUP_GAP_FRAC · max_pressure,
    eta ≤ max_eta, sun-collision, and a still-mine arrival heuristic
    (destination not about to flip from queued attacks within eta).
    """
    out = []
    if not pressure:
        return out
    max_p = max(pressure.values()) if pressure else 0.0
    if max_p <= 0:
        return out
    gap_threshold = REGROUP_GAP_FRAC * max_p

    sources = sorted(
        [m for m in mine if ships_left.get(m.id, 0) >= min_send],
        key=lambda m: ships_left.get(m.id, 0), reverse=True,
    )
    dests = sorted(mine, key=lambda m: pressure.get(m.id, 0.0), reverse=True)
    used_dests = set()
    for s in sources:
        s_p = pressure.get(s.id, 0.0)
        for d in dests:
            if d.id == s.id or d.id in used_dests:
                continue
            d_p = pressure.get(d.id, 0.0)
            gap = d_p - s_p
            if gap <= gap_threshold:
                # dests are pressure-sorted desc, so once gap fails, no later dst works
                break
            send = min(ships_left[s.id], int(s.ships))
            if send < min_send:
                break
            # arrival check
            if d.id in moving_planets:
                angle, arrive_tick = find_angle_to_moving_planet(s, d, send, vel)
                if angle is None or arrive_tick is None or arrive_tick > max_eta:
                    continue
            else:
                dist = math.sqrt((s.x - d.x) ** 2 + (s.y - d.y) ** 2)
                speed = _planet_fleet_speed(send)
                eta = dist / max(speed, 1e-6)
                if eta > max_eta:
                    continue
                angle = math.atan2(d.y - s.y, d.x - s.x)
                if sun_collision(s, speed, angle):
                    continue
                arrive_tick = math.floor(eta)
            # still-mine: skip if d would fall from queued attacks within arrive_tick
            if d.id in under_attack:
                cum_atk = 0
                losing = False
                for att in sorted(under_attack[d.id]["fleets"], key=lambda a: a["arrive_tick"]):
                    at_t = att["arrive_tick"]
                    if at_t > arrive_tick + 1:
                        break
                    cum_atk += att["fleet"].ships
                    if cum_atk > d.ships + at_t * d.production:
                        losing = True
                        break
                if losing:
                    continue
            out.append([s.id, angle, send])
            used_dests.add(d.id)
            ships_left[s.id] -= send
            break
    return out


def agent(obs):
    global steps
    global fleet_trajectories
    global reinforcement_trajectories
    moves = []
    
    if steps < 2:
        steps += 1
        return []
    if steps == 2:
        fill_moving_planets(obs)
        steps = 3  

    lobs = refresh_local_obs(obs)
    update_fleet_trajectories(lobs.get("fleets", []))
    update_reinforcement_trajectories(lobs.get("planets", []))
    comet_planet_ids = obs.get("comet_planet_ids", [])
    under_attack = get_planets_under_attack(lobs.get("mine", []), lobs.get("fleets", []), lobs.get("player", -2), obs.angular_velocity)
    exhausted_planets_id = set()

    if not lobs.get("targets", []):
        return []

    num_players = max(2, max((p.owner for p in lobs["planets"] if p.owner >= 0), default=1) + 1)
    step_now = obs.get("step", steps)

    # v138 precompute: safe_drain per source + enemy pressure per owned planet.
    mine_list = lobs.get("mine", [])
    enemy_planets = [
        p for p in lobs.get("planets", [])
        if p.owner >= 0 and p.owner != lobs.get("player", -2) and p.ships > 0
    ]
    safe_drain_by_id = {
        m.id: safe_drain_for_source(m, under_attack, reinforcement_trajectories)
        for m in mine_list
    }
    enemy_pressure = compute_enemy_pressure(mine_list, enemy_planets)
    effective_min_ships = MIN_SHIPS_MINE_ATTACK

    reinforcement_plans = get_reinforcement_plans(lobs.get("mine", []), under_attack)
    for p, plan in reinforcement_plans.items():
        already_reinforced = any(
            r["target"].id == p.id and r["arrive_tick"] >= 0
            for r in reinforcement_trajectories
        )

        if already_reinforced:
            continue
            
        ships_needed = plan["ships_needed"]
        needed_by_tick = plan["needed_by_tick"]
        nearest_planets = get_closest_planets_to_target(lobs.get("mine", []), p)
        
        for row in nearest_planets:
            p_np, _ = row
            
            if p_np.id == p.id or p_np.id in exhausted_planets_id:
                continue

            p_np_available_ships = p_np.ships

            reserved_reinforcement_ships = sum(
                r["total_ships"]
                for r in reinforcement_trajectories
                if r["mine"].id == p_np.id
            )
            
            p_np_available_ships -= reserved_reinforcement_ships

            if p_np.id in under_attack:
                enemy_ships = sum(
                    att["fleet"].ships
                    for att in under_attack[p_np.id]["fleets"]
                )
                p_np_available_ships = max(0, p_np_available_ships - enemy_ships)

            sent_reinforcements = max(effective_min_ships, ships_needed)

            if p_np_available_ships < sent_reinforcements:
                continue
            angle_np = None
            if p.id not in moving_planets:
                angle_np = math.atan2(p.y - p_np.y, p.x - p_np.x)
                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(sent_reinforcements) / math.log(1000)) ** 1.5                
                dist = math.sqrt((p.x - p_np.x)**2 + (p.y - p_np.y)**2)
                arrive_tick = math.floor(dist / fleet_speed)

                if arrive_tick > needed_by_tick:
                    continue
                
            else:
                angle_np, arrive_tick = find_angle_to_moving_planet(p_np, p, sent_reinforcements, obs.angular_velocity)

            if angle_np is None or arrive_tick is None:
                continue
            
            moves.append([p_np.id, angle_np, sent_reinforcements])
            exhausted_planets_id.add(p_np.id)
            reinforcement_trajectories.append({
                "mine": p_np,
                "target": p,
                "angle": angle_np,
                "total_ships": sent_reinforcements,
                "arrive_tick": arrive_tick
            })
            break

    for m in sorted(lobs.get("mine", []), key=lambda p: p.ships, reverse=True):
        if m.id in exhausted_planets_id:
            continue

        if m.ships < effective_min_ships:
            continue

        candidate_targets = []
        for t in lobs.get("targets", []):
            if t.id in comet_planet_ids:
                continue

            score = get_custom_score(m, t)
            candidate_targets.append((m, t, score))

        candidate_targets = sorted(candidate_targets, key=lambda x: x[2], reverse=True)
        
    
        for m, t, s in candidate_targets[:3]:
            # v138 C: safe_drain replaces "ships − current attackers" heuristic.
            m_available_ships = safe_drain_by_id.get(m.id, int(m.ships))

            if m_available_ships < effective_min_ships:
                continue

            nearest_planets = get_closest_planets_to_target(lobs.get("mine", []), t)
            safe_nearest_planets = []
            for p, dist in nearest_planets: # check which planets are fit to attack and are not vulnerable
                if p.id == m.id or p.id in exhausted_planets_id:
                    continue

                available_ships = safe_drain_by_id.get(p.id, int(p.ships))

                if available_ships < effective_min_ships:
                    continue
                
                safe_nearest_planets.append((p, dist, available_ships))
            
            owned_count = len(lobs.get("mine", []))
            total_count = len(lobs.get("planets", []))
    
            en_route = 0
            if fleet_trajectories:
                en_route = sum(
                    f["total_ships"]
                    for f in fleet_trajectories
                    if f["target"].id == t.id
                )
    
            needed_now = t.ships + 1
            if t.owner != -1:
                needed_now += 3 * t.production
            
            if owned_count < total_count * 0.75: # release all havoc when targets less than ~25%
                if en_route >= needed_now:
                    continue
            
            base_ships = max(effective_min_ships, needed_now - en_route)
            
            extra_ships = 0
            fleet_speed = 0
            angle = None
            arrive_tick = None
    
            if m_available_ships >= base_ships: # single attack
                send_ships = base_ships
                if num_players == 2 and m.id not in under_attack and m_available_ships >= 40:
                    send_ships = m_available_ships

                if t.id in moving_planets: # single moving planet
                    total_ships = send_ships
                    
                    for _ in range(3):
                        angle, arrive_tick = find_angle_to_moving_planet(m, t, total_ships, obs.angular_velocity)

                        if angle is None:
                            break

                        if t.owner != -1:
                            new_total_ships = base_ships + arrive_tick * t.production
                        else:
                            new_total_ships = base_ships

                        if new_total_ships > m_available_ships:
                            angle = None
                            break
                        
                        if new_total_ships == total_ships:
                            break

                        total_ships = new_total_ships
                    extra_ships = total_ships - base_ships
                        
                else: # single static planet
                    angle = calculate_angle(m, t) # single static unowned
                    total_ships = send_ships
                    dist = math.sqrt((t.x - m.x)**2 + (t.y - m.y)**2)
                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5
                    arrive_tick = math.floor(dist / fleet_speed)
                    
                    if t.owner != -1: # single static owned
                        for _ in range(3):
                            fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5
                            turns_to_arrive = math.floor(dist / fleet_speed)
                            
                            extra_ships = turns_to_arrive * t.production
                            new_total_ships = base_ships + extra_ships

                            if new_total_ships > m_available_ships:
                                angle = None
                                arrive_tick = None
                                break

                            arrive_tick = turns_to_arrive
                            
                            if new_total_ships == total_ships:
                                break
                            
                            total_ships = new_total_ships

                        extra_ships = total_ships - base_ships
                        
                if angle is not None and arrive_tick is not None:
                    fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, total_ships)) / math.log(1000)) ** 1.5

                    collides_sun = sun_collision(m, fleet_speed, angle)
                    if collides_sun:
                        continue
                        
                    moves.append([m.id, angle, total_ships])
                    exhausted_planets_id.add(m.id)
                    fleet_trajectories.append({
                        "mine": m,
                        "target": t,
                        "angle": angle,
                        "total_ships": total_ships,
                        "arrive_tick": arrive_tick
                    })
            
            elif m_available_ships < base_ships and len(lobs.get("mine", [])) > 1 and t.ships >= MIN_SHIPS_TARGET_COOP_ATTACK: # coop attack
                accum = m_available_ships
                attacking_planets = [{"planet": m, "ships": m_available_ships}]
                coop_sent = False
                
                for p, dist, p_available_ships in safe_nearest_planets:
                    if coop_sent:
                        break
                    
                    attacking_planets.append({"planet": p, "ships": p_available_ships})
                    accum += p_available_ships
    
                    if len(attacking_planets) > COOP_PLANET_CAP:
                        break
                    
                    if accum < base_ships:
                        continue
                        
                    if t.id not in moving_planets: # coop static planet
                        if t.owner == -1: # coop static unowned
                            remainder = base_ships
                            planned = []
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
    
                                if p_ships <= 0:
                                    continue
                                
                                angle = calculate_angle(p, t)
                                dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(p_ships) / math.log(1000)) ** 1.5
                                arrive_tick = math.floor(dist / fleet_speed)
                                
                                collides_sun = sun_collision(p, fleet_speed=fleet_speed, angle=angle)
                                if collides_sun:
                                    break
    
                                remainder -= p_ships
                                    
                                planned.append([p, angle, p_ships, arrive_tick])
    
                            if remainder > 0:
                                continue
                                
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
                                
                        else: # coop static owned
                            required_ships = calculate_req_ships(attacking_planets, t, base_ships)
                            remainder = required_ships
                            
                            if accum < required_ships: 
                                continue
                                
                            planned = []
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
    
                                if p_ships <= 0:
                                    continue
                                    
                                angle = calculate_angle(p, t)
                                dist = math.sqrt((p.x - t.x)**2 + (p.y - t.y)**2)
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(p_ships) / math.log(1000)) ** 1.5
                                arrive_tick = math.floor(dist / fleet_speed)
                                
                                collides_sun = sun_collision(p, fleet_speed=fleet_speed, angle=angle)
                                if collides_sun:
                                    continue
    
                                remainder -= p_ships
                                
                                planned.append([p, angle, p_ships, arrive_tick])
    
                            if remainder > 0:
                                continue
                            
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
                    
                    else: # coop moving planet
                        planet_trajectories = get_planet_trajectories(t, obs.angular_velocity)
                        if t.owner == -1: # coop moving unowned
                            remainder = base_ships
                            planned = []
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))
    
                                if p_ships <= 0:
                                    continue
    
                                angle, arrive_tick = find_angle_to_moving_planet(p, t, p_ships, obs.angular_velocity)
                                
                                if angle is None or arrive_tick is None:
                                    continue
                                    
                                planned.append([p, angle, p_ships, arrive_tick])
                                remainder -= p_ships
    
                            if remainder > 0:
                                continue
    
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)
    
                            coop_sent = True
                            break
                    
                        else: # coop moving owned
                            required_ships = calculate_req_ships_moving(attacking_planets, t, base_ships, obs.angular_velocity)
                            remainder = required_ships
                            planned = []
    
                            if accum < required_ships:
                                continue
                            
                            for a_p in attacking_planets:
                                p = a_p["planet"]
                                p_ships = min(a_p["ships"], remainder)
                                
                                if p_ships > 0:
                                    p_ships = min(a_p["ships"], max(p_ships, MIN_SHIPS_MINE_ATTACK))   
                                    
                                if p_ships <= 0:
                                    continue
                                
                                fleet_speed = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, p_ships)) / math.log(1000)) ** 1.5
    
                                angle, arrive_tick = find_angle_to_moving_planet(p, t, p_ships, obs.angular_velocity)
    
                                if angle is None or arrive_tick is None:
                                    continue
                                
                                remainder -= p_ships
    
                                planned.append([p, angle, p_ships, arrive_tick])
    
                            if remainder > 0:
                                continue
                            
                            for move in planned:
                                fleet_trajectories.append({
                                    "mine": move[0],
                                    "target": t,
                                    "angle": move[1],
                                    "total_ships": move[2],
                                    "arrive_tick": move[3]
                                })
                                exhausted_planets_id.add(move[0].id)
                                move[0] = move[0].id
                                moves.append(move)

                            coop_sent = True
                            break

    # v138 A: always-on pressure-gradient regroup with leftover ships.
    # Recompute leftover from the do-nothing safe_drain cap minus what was sent.
    sent_per_planet = {}
    for mv in moves:
        sent_per_planet[mv[0]] = sent_per_planet.get(mv[0], 0) + mv[2]
    # Role mutex: a planet that already sent or already received reinforcement
    # this turn should not double up. acting_sources = anyone who sent; received
    # = reinforcement targets + previously regroup-targeted (tracked here).
    acting_sources = set(sent_per_planet.keys())
    received_set = {p.id for p in reinforcement_plans.keys()} if reinforcement_plans else set()
    leftover = {}
    for m in mine_list:
        if m.id in received_set:
            leftover[m.id] = 0          # planet getting a defensive top-up — don't drain
            continue
        cap = safe_drain_by_id.get(m.id, int(m.ships))
        leftover[m.id] = max(0, cap - sent_per_planet.get(m.id, 0))
    regroup_moves = plan_regroup(
        mine_list, leftover, enemy_pressure, under_attack, obs.angular_velocity,
    )
    for rm in regroup_moves:
        src_id, angle, ships = rm
        if src_id in acting_sources:
            continue   # role mutex
        moves.append(rm)
        # Track as reinforcement so next turn's planning sees it as in-flight aid.
        src_planet = next((p for p in mine_list if p.id == src_id), None)
        if src_planet is None:
            continue
        dx_best = float("inf")
        dst_planet = None
        for d in mine_list:
            if d.id == src_id:
                continue
            d_ang = math.atan2(d.y - src_planet.y, d.x - src_planet.x)
            err = abs((d_ang - angle + math.pi) % (2 * math.pi) - math.pi)
            if err < dx_best:
                dx_best = err
                dst_planet = d
        if dst_planet is None:
            continue
        dist = math.sqrt((dst_planet.x - src_planet.x) ** 2 + (dst_planet.y - src_planet.y) ** 2)
        fleet_speed_est = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(1, ships)) / math.log(1000)) ** 1.5
        arrive_tick = math.floor(dist / max(fleet_speed_est, 1e-6))
        reinforcement_trajectories.append({
            "mine": src_planet,
            "target": dst_planet,
            "angle": angle,
            "total_ships": ships,
            "arrive_tick": arrive_tick,
        })

    return moves