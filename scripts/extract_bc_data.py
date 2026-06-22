"""
Extract (state, action) pairs from Kaggle replays for behavioral cloning.

For each step in each replay, for each planet owned by an opponent:
  - Features: planet state, game context, threat level
  - Label: did they launch? if so, target type + ship fraction

Output: data/bc_training.jsonl — one line per (planet, step) sample.

Usage:
    .venv/bin/python scripts/extract_bc_data.py
"""
import json, glob, os, math, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPLAY_DIRS = [
    os.path.join(ROOT, 'replays', '2p'),
    os.path.join(ROOT, 'replays', '4p'),
    '/tmp/orbit_analysis/replays',
    '/tmp/orbit_new',
]
OUT_FILE = os.path.join(ROOT, 'data', 'bc_training.jsonl')

# Load leaderboard for tier labels
lb = {}
lb_file = '/tmp/orbit_analysis/leaderboard.json'
if os.path.exists(lb_file):
    lb = json.load(open(lb_file))

CENTER = 50.0
SUN_R = 10.0


def tier_label(score):
    if score >= 1000: return '1000+'
    if score >= 900: return '900'
    if score >= 800: return '800'
    if score >= 700: return '700'
    return '600'


def find_target(planets, src_x, src_y, angle, owner_id):
    """Project fleet trajectory to find likely target planet."""
    speed = 3.0  # approximate
    best_pid = -1
    best_dist = 999
    best_type = 'unknown'

    for tick in range(1, 40):
        fx = src_x + math.cos(angle) * speed * tick
        fy = src_y + math.sin(angle) * speed * tick
        # out of bounds
        if fx < 0 or fx > 100 or fy < 0 or fy > 100:
            break
        # sun
        if math.hypot(fx - CENTER, fy - CENTER) < SUN_R:
            break

        for p in planets:
            pid, powner, px, py, pradius = p[0], p[1], p[2], p[3], p[4]
            d = math.hypot(px - fx, py - fy)
            if d < pradius + 2 and d < best_dist:
                best_dist = d
                best_pid = pid
                if powner == owner_id:
                    best_type = 'reinforce'
                elif powner == -1:
                    best_type = 'neutral'
                else:
                    best_type = 'enemy'
        if best_pid >= 0:
            break

    return best_pid, best_type


def extract_replay(fpath):
    """Extract all training samples from one replay."""
    try:
        data = json.load(open(fpath))
    except:
        return []

    info = data.get('info', {})
    teams = info.get('TeamNames', [])
    rewards = data.get('rewards', [])
    steps = data.get('steps', [])
    n_players = len(rewards)

    if n_players < 2 or len(steps) < 20:
        return []

    samples = []

    for player_idx in range(n_players):
        name = teams[player_idx] if player_idx < len(teams) else ''
        if 'Borrun' in name:
            continue

        score = lb.get(name, 0)
        if score < 600:
            continue
        tier = tier_label(score)

        prev_fleet_ids = set()

        for si in range(0, min(len(steps), 450), 2):  # sample every 2 steps
            step = steps[si]
            if not isinstance(step, list) or player_idx >= len(step):
                continue

            obs = step[player_idx].get('observation', {})
            if not isinstance(obs, dict):
                continue

            planets = obs.get('planets', [])
            fleets = obs.get('fleets', [])
            player_id = obs.get('player', player_idx)

            my_planets = [p for p in planets if p[1] == player_id]
            enemy_planets = [p for p in planets if p[1] != player_id and p[1] != -1]
            neutral_planets = [p for p in planets if p[1] == -1]

            if not my_planets:
                continue

            # Global features
            my_total_ships = sum(p[5] for p in my_planets)
            my_fleet_ships = sum(f[6] for f in fleets if f[1] == player_id)
            enemy_total_ships = sum(p[5] for p in enemy_planets) + sum(
                f[6] for f in fleets if f[1] != player_id and f[1] != -1)
            my_prod = sum(p[6] for p in my_planets)
            enemy_prod = sum(p[6] for p in enemy_planets)
            total_planets = len(planets)
            my_count = len(my_planets)
            enemy_count = len(enemy_planets)
            neutral_count = len(neutral_planets)

            power = my_count / max(1, my_count + enemy_count)
            ship_ratio = (my_total_ships + my_fleet_ships) / max(
                1, my_total_ships + my_fleet_ships + enemy_total_ships)

            # Detect new launches by this player
            cur_fleet_ids = {}
            for f in fleets:
                if f[1] == player_id:
                    cur_fleet_ids[f[0]] = f

            new_launches = {}
            for fid, f in cur_fleet_ids.items():
                if fid not in prev_fleet_ids:
                    from_pid = f[5]  # from_planet_id
                    new_launches.setdefault(from_pid, []).append(f)

            prev_fleet_ids = set(cur_fleet_ids.keys())

            # Per-planet samples
            for mp in my_planets:
                pid = mp[0]
                mx, my_y = mp[2], mp[3]
                mships = mp[5]
                mprod = mp[6]

                # Nearest enemy
                if enemy_planets:
                    ne = min(enemy_planets, key=lambda p: math.hypot(p[2]-mx, p[3]-my_y))
                    nearest_enemy_dist = math.hypot(ne[2]-mx, ne[3]-my_y)
                    nearest_enemy_ships = ne[5]
                else:
                    nearest_enemy_dist = 100.0
                    nearest_enemy_ships = 0

                # Nearest neutral
                if neutral_planets:
                    nn = min(neutral_planets, key=lambda p: math.hypot(p[2]-mx, p[3]-my_y))
                    nearest_neutral_dist = math.hypot(nn[2]-mx, nn[3]-my_y)
                    nearest_neutral_ships = nn[5]
                else:
                    nearest_neutral_dist = 100.0
                    nearest_neutral_ships = 0

                # Incoming threat
                threat_ships = 0
                for f in fleets:
                    if f[1] == player_id:
                        continue
                    fx, fy, fa = f[2], f[3], f[4]
                    fspeed = 1.0 + 5.0 * (math.log(max(1, f[6])) / math.log(1000)) ** 1.5
                    for tick in range(1, 20):
                        tx = fx + math.cos(fa) * fspeed * tick
                        ty = fy + math.sin(fa) * fspeed * tick
                        if math.hypot(tx - mx, ty - my_y) < mp[4] + 2:
                            threat_ships += f[6]
                            break

                # Did this planet launch?
                launched = pid in new_launches
                if launched:
                    launch_fleets = new_launches[pid]
                    total_launched = sum(f[6] for f in launch_fleets)
                    ship_frac = total_launched / max(1, mships)
                    # Find target type from first fleet
                    f0 = launch_fleets[0]
                    _, tgt_type = find_target(planets, f0[2], f0[3], f0[4], player_id)
                else:
                    total_launched = 0
                    ship_frac = 0.0
                    tgt_type = 'none'

                sample = {
                    'tier': tier,
                    'step': si,
                    'n_players': n_players,
                    # Planet features
                    'ships': mships,
                    'production': mprod,
                    'dist_center': math.hypot(mx - CENTER, my_y - CENTER),
                    # Context
                    'my_planets': my_count,
                    'enemy_planets': enemy_count,
                    'neutral_planets': neutral_count,
                    'power': round(power, 3),
                    'ship_ratio': round(ship_ratio, 3),
                    'my_prod': my_prod,
                    'enemy_prod': enemy_prod,
                    # Local threat/opportunity
                    'nearest_enemy_dist': round(nearest_enemy_dist, 1),
                    'nearest_enemy_ships': nearest_enemy_ships,
                    'nearest_neutral_dist': round(nearest_neutral_dist, 1),
                    'nearest_neutral_ships': nearest_neutral_ships,
                    'threat_ships': threat_ships,
                    # Labels
                    'launched': int(launched),
                    'ship_frac': round(ship_frac, 3),
                    'tgt_type': tgt_type,  # none/neutral/enemy/reinforce
                }
                samples.append(sample)

    return samples


# Process all replays
all_replays = set()
for d in REPLAY_DIRS:
    for fpath in glob.glob(os.path.join(d, 'episode-*-replay.json')):
        all_replays.add(fpath)

print(f"Found {len(all_replays)} replay files")

total_samples = 0
tier_counts = defaultdict(int)

with open(OUT_FILE, 'w') as out:
    for i, fpath in enumerate(sorted(all_replays)):
        samples = extract_replay(fpath)
        for s in samples:
            out.write(json.dumps(s) + '\n')
            tier_counts[s['tier']] += 1
        total_samples += len(samples)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(all_replays)} replays, {total_samples} samples")

print(f"\nDone: {total_samples} samples -> {OUT_FILE}")
print(f"Per tier: {dict(tier_counts)}")
