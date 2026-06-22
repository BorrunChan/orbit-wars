"""
Field bench — sample opponents from real Kaggle profiles by rating tier.

Uses opponent_styles.jsonl (1028 profiles) to generate parametric agents
that match each tier's actual behavior distribution. Simulates a climb
from 600 by matching against opponents at various tiers.

Usage:
    .venv/bin/python scripts/bench_field.py agents/v132_situational.py [n_per_tier]
"""
import sys, os, json, math, random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments import make

AGENT = sys.argv[1] if len(sys.argv) > 1 else 'main.py'
N_PER_TIER = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# Load profiles
STYLES = [json.loads(l) for l in open('data/opponent_styles.jsonl')]
STYLES = [s for s in STYLES if s.get('lb_score') and s['lb_score'] > 0]

TIERS = [
    ('600-700',  600,  700),
    ('700-800',  700,  800),
    ('800-850',  800,  850),
    ('850-900',  850,  900),
    ('900-1000', 900, 1000),
    ('1000+',   1000, 9999),
]

CENTER = 50.0
SUN_RADIUS = 10.0
SUN_BUFFER = 0.6

def _safe_angle(src, dst):
    sx, sy = src; tx, ty = dst
    tdx, tdy = tx-sx, ty-sy
    tdist = math.hypot(tdx, tdy)
    if tdist < 1e-6: return 0.0
    tgt_a = math.atan2(tdy, tdx)
    sdx, sdy = CENTER-sx, CENTER-sy
    sdist = math.hypot(sdx, sdy)
    R = SUN_RADIUS + SUN_BUFFER
    if sdist <= R: return tgt_a
    sun_a = math.atan2(sdy, sdx)
    sun_h = math.asin(min(1.0, R/sdist))
    diff = (tgt_a - sun_a + math.pi) % (2*math.pi) - math.pi
    if abs(diff) >= sun_h + 0.05: return tgt_a
    return sun_a + (sun_h + 0.05 if diff >= 0 else -sun_h - 0.05)


def make_profile_agent(profile, seed=42):
    """Generate an agent function from an opponent_styles profile."""
    rng = random.Random(seed)
    lr = profile.get('launch_rate', 0.5)
    avg_fl = max(5, profile.get('avg_fleet', 20))
    std_fl = max(1, avg_fl * 0.4)

    def agent_fn(obs):
        if isinstance(obs, dict):
            getf = obs.get
        else:
            getf = lambda k, d=None: getattr(obs, k, d)
        player = getf("player", 0)
        planets = getf("planets", []) or []
        my_planets = [p for p in planets if p[1] == player]
        if not my_planets:
            return []
        neutrals = [p for p in planets if p[1] == -1]
        enemies = [p for p in planets if p[1] != player and p[1] != -1]

        actions = []
        for mp in my_planets:
            if mp[5] < 4:
                continue
            p_launch = min(1.0, lr / max(1, len(my_planets)))
            if rng.random() > p_launch:
                continue
            if rng.random() < 0.4 and neutrals:
                tgts = neutrals
            elif enemies:
                tgts = enemies
            elif neutrals:
                tgts = neutrals
            else:
                continue
            tgt = min(tgts, key=lambda q: math.hypot(q[2]-mp[2], q[3]-mp[3]))
            ships_want = max(4, int(rng.gauss(avg_fl, std_fl)))
            ships = min(ships_want, max(1, mp[5] - 1))
            if ships < 4:
                continue
            angle = _safe_angle((mp[2], mp[3]), (tgt[2], tgt[3]))
            actions.append([mp[0], angle, ships])
        return actions
    return agent_fn


def run_game(agent_path, opp_fn, seed):
    try:
        env = make('orbit_wars', configuration={'seed': seed}, debug=True)
        env.run([agent_path, opp_fn])
        last = env.steps[-1]
        won = last[0].reward > last[1].reward
        return won, len(env.steps)
    except Exception as e:
        return None, 0


rng = random.Random(12345)
print(f"Field bench: {AGENT}")
print(f"Sampling {N_PER_TIER} opponents per tier from {len(STYLES)} profiles")
print("=" * 80)

total_w, total_g = 0, 0

for tier_label, lo, hi in TIERS:
    pool = [s for s in STYLES if lo <= s['lb_score'] < hi]
    if not pool:
        print(f"\n{tier_label}: no profiles")
        continue

    sampled = rng.sample(pool, min(N_PER_TIER, len(pool)))
    wins, games = 0, 0

    print(f"\n--- {tier_label} ({len(pool)} profiles, sampled {len(sampled)}) ---")

    for profile in sampled:
        opp_name = profile['opp_name'][:25]
        opp_lr = profile['launch_rate']
        opp_fl = profile.get('avg_fleet', 20)
        opp_lb = profile['lb_score']

        opp_fn = make_profile_agent(profile, seed=hash(opp_name) & 0xFFFF)
        seed = rng.randint(1, 100000)
        won, steps = run_game(AGENT, opp_fn, seed)

        if won is None:
            print(f"  ERROR vs {opp_name}")
            continue

        wins += int(won)
        games += 1
        status = 'W' if won else 'L'
        print(f"  {status} vs {opp_name:<25} lb={opp_lb:>6.0f} lr={opp_lr:.2f} fl={opp_fl:.0f} ({steps} steps)")

    if games > 0:
        wr = wins / games * 100
        print(f"  => {tier_label}: {wins}/{games} = {wr:.0f}%")
        total_w += wins
        total_g += games

print(f"\n{'='*80}")
print(f"Total: {total_w}/{total_g} = {total_w/total_g*100:.1f}%")
