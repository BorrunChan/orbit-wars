"""
Kaggle climb simulator — TrueSkill-based rating simulation.

Simulates the Kaggle climb from μ=600: matchmakes against an opponent pool
with assigned ratings, plays 2P/4P games, updates TrueSkill, reports
convergence rating.

Opponent pool: real notebook bots + tiered structured + bc_smart, each
assigned a fixed rating based on their estimated strength.

Usage:
    .venv/bin/python scripts/climb_sim.py agents/v135_adaptive_min.py [n_games]
"""
import sys, os, random, math, importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trueskill
from kaggle_environments import make

AGENT = sys.argv[1] if len(sys.argv) > 1 else 'main.py'
N_GAMES = int(sys.argv[2]) if len(sys.argv) > 2 else 100
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 42

# TrueSkill environment matching Kaggle defaults
env_ts = trueskill.TrueSkill(mu=600, sigma=200, beta=100, tau=3, draw_probability=0)

# Opponent pool with estimated ratings
# Format: (path_or_callable, name, rating, is_callable)
# Calibrated ratings: v128 should converge ~890 to match Kaggle reality.
# Previous ratings were too low (v128 converged 978 vs real 890).
# Raise all by ~100 and add more strong opponents.
OPPONENT_POOL = [
    # ~750-800 tier
    ('opponents/tactical2026.py',      'tactical2026',    750),
    ('opponents/kaggle_low.py',        'kaggle_low',      800),
    ('opponents/structured_tier700.py','struct_700',      810),
    ('opponents/kaggle_high2.py',      'kaggle_high2',    800),

    # ~850 tier
    ('opponents/kaggle_mid.py',        'kaggle_mid',      850),
    ('opponents/smart_swarmer_v3.py',  'smart_swarmer',   840),
    ('opponents/kaggle_high.py',       'kaggle_high',     860),
    ('opponents/peak1103.py',          'peak1103',        870),

    # ~900 tier
    ('opponents/proto1000.py',         'proto1000',       900),
    ('opponents/structured.py',        'structured',      920),
    ('opponents/reinforce958.py',      'reinforce958',    910),

    # ~950-1000 tier
    ('opponents/structured_tier900.py','struct_900',      950),
    ('opponents/lb1224.py',            'lb1224',          970),
    ('opponents/orbitbotnext.py',      'orbitbotnext',    980),
    ('opponents/sundodge.py',          'sundodge',        960),

    # ~1100+ tier
    ('opponents/structured_tier1100.py','struct_1100',    1100),
    ('opponents/mlhybrid.py',          'mlhybrid',        1050),
]

BC_RATINGS = {'800': 880, '900': 940, '1000+': 1020}

rng = random.Random(SEED)


def make_bc_agent(tier):
    os.environ['BC_TIER'] = tier
    import opponents.bc_smart as bcs
    importlib.reload(bcs)
    bcs.moving_planets.clear()
    bcs.fleet_trajectories.clear()
    bcs.reinforcement_trajectories.clear()
    bcs.steps = 0
    return 'opponents/bc_smart.py'


def pick_opponents(my_mu, n_players, rng):
    """Pick opponents close to our rating, with some variance."""
    candidates = []
    for path, name, rating in OPPONENT_POOL:
        dist = abs(rating - my_mu)
        # Weight inversely proportional to distance, but allow some range
        weight = max(0.1, 1.0 / (1.0 + dist / 150))
        candidates.append((path, name, rating, weight))

    # Add BC-smart opponents
    for tier, rating in BC_RATINGS.items():
        dist = abs(rating - my_mu)
        weight = max(0.1, 1.0 / (1.0 + dist / 150))
        candidates.append((f'bc_smart:{tier}', f'bc_{tier}', rating, weight))

    # Weighted sample
    total_weight = sum(w for _, _, _, w in candidates)
    selected = []
    for _ in range(n_players - 1):
        r = rng.random() * total_weight
        cumul = 0
        for path, name, rating, weight in candidates:
            cumul += weight
            if cumul >= r:
                selected.append((path, name, rating))
                break

    return selected


def run_game(agent_path, opponents, game_seed):
    """Run a 2P or 4P game. Returns list of (player_idx, reward) sorted."""
    n = len(opponents) + 1

    # Prepare opponent agents
    opp_agents = []
    for path, name, rating in opponents:
        if path.startswith('bc_smart:'):
            tier = path.split(':')[1]
            opp_agents.append(make_bc_agent(tier))
        else:
            opp_agents.append(path)

    agents = [agent_path] + opp_agents

    try:
        env = make('orbit_wars', configuration={'seed': game_seed}, debug=True)
        env.run(agents)
        last = env.steps[-1]
        rewards = [last[i].reward for i in range(n)]
        return rewards, len(env.steps)
    except Exception as e:
        return None, 0


# Initialize our rating
our_rating = env_ts.create_rating()

print(f"Climb simulation: {AGENT}")
print(f"Starting μ={our_rating.mu:.0f}, σ={our_rating.sigma:.0f}")
print(f"Opponent pool: {len(OPPONENT_POOL)} bots + {len(BC_RATINGS)} BC-smart")
print(f"Games: {N_GAMES} (2P/4P mixed)")
print("=" * 80)

trajectory = []
wins_2p, losses_2p = 0, 0
r4p = [0, 0, 0, 0]

for game_i in range(N_GAMES):
    # Decide 2P or 4P (50/50)
    n_players = 2 if rng.random() < 0.5 else 4

    # Pick opponents
    opponents = pick_opponents(our_rating.mu, n_players, rng)

    # Play
    game_seed = rng.randint(1, 999999)
    rewards, steps = run_game(AGENT, opponents, game_seed)

    if rewards is None:
        print(f"  Game {game_i+1}: ERROR")
        continue

    # TrueSkill update
    opp_ratings = [env_ts.create_rating(mu=r, sigma=50) for _, _, r in opponents]
    all_ratings = [our_rating] + opp_ratings

    # Create rating groups and rank order
    rating_groups = tuple(({i: all_ratings[i]},) for i in range(n_players))
    # Flatten: each group is a dict
    groups = [{0: our_rating}]
    for i, (_, _, r) in enumerate(opponents):
        groups.append({i+1: opp_ratings[i]})

    # Rank by reward (higher = better rank = lower rank number)
    ranked_indices = sorted(range(n_players), key=lambda x: -rewards[x])
    ranks = [0] * n_players
    for rank, idx in enumerate(ranked_indices):
        ranks[idx] = rank

    # Update using trueskill
    groups_for_ts = [({0: our_rating},)] if False else None
    # Simpler: just update our rating based on outcome
    if n_players == 2:
        opp_r = opp_ratings[0]
        if rewards[0] > rewards[1]:
            our_rating, opp_r = trueskill.rate_1vs1(our_rating, opp_r, env=env_ts)
            result = 'WIN'
            wins_2p += 1
        else:
            opp_r, our_rating = trueskill.rate_1vs1(opp_r, our_rating, env=env_ts)
            result = 'LOSS'
            losses_2p += 1
    else:
        # 4P: use rate() with ranking
        groups = [{0: our_rating}]
        for i in range(len(opponents)):
            groups.append({0: opp_ratings[i]})

        ranked = sorted(range(n_players), key=lambda x: -rewards[x])
        rank_order = [0] * n_players
        for pos, idx in enumerate(ranked):
            rank_order[idx] = pos

        rated = env_ts.rate(groups, ranks=rank_order)
        our_rating = rated[0][0]
        rank = rank_order[0] + 1
        r4p[rank-1] += 1
        result = f'R{rank}'

    opp_str = ', '.join(f'{n}({r:.0f})' for _, n, r in opponents)
    trajectory.append((game_i+1, our_rating.mu, our_rating.sigma, result))

    if (game_i + 1) % 10 == 0 or game_i < 5:
        print(f"  Game {game_i+1:>3}: μ={our_rating.mu:>6.0f} σ={our_rating.sigma:>5.0f} {result:>4} vs [{opp_str}]")

# Summary
print(f"\n{'='*80}")
print(f"Final: μ={our_rating.mu:.0f}, σ={our_rating.sigma:.0f}")
print(f"2P: {wins_2p}W/{losses_2p}L = {wins_2p/(wins_2p+losses_2p)*100:.0f}%" if wins_2p+losses_2p else "")
t4 = sum(r4p)
if t4:
    print(f"4P: R1={r4p[0]} R2={r4p[1]} R3={r4p[2]} R4={r4p[3]} (R1={r4p[0]/t4*100:.0f}% R4={r4p[3]/t4*100:.0f}%)")

# Trajectory summary
print(f"\nTrajectory (every 10 games):")
for i, mu, sigma, result in trajectory:
    if i % 10 == 0 or i <= 3:
        print(f"  Game {i:>3}: μ={mu:>6.0f}")
