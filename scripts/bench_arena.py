"""
Arena bench — full field simulation with architecture diversity × tier weighting.

Real Kaggle field composition (850-1000 segment):
  48% precise_structured (sim-greedy framework: structured/lb1224/orbitbotnext)
  24% moderate_pressure  (proto1000-style: coop + reinforce + formula scoring)
  14% high_volume_swarm  (high-frequency small fleets)
  14% passive/defensive  (low activity, big fleet when they do fire)

Each archetype has multiple implementations at different strength levels.
Weighted scoring reflects the real opponent distribution.

Usage:
    .venv/bin/python scripts/bench_arena.py agents/v131_climb.py [seeds_per_matchup]
"""
import sys, os, importlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kaggle_environments import make

AGENT = sys.argv[1] if len(sys.argv) > 1 else 'main.py'
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# Full opponent pool with archetype labels and field weights
# weight = how many "copies" of this opponent in the weighted average
# (reflects how common this type is in real Kaggle)
POOL = [
    # === PRECISE_STRUCTURED (48% of field) ===
    ('opponents/structured_tier700.py',  'precise', 2, 'sim-greedy weak (700-level)'),
    ('opponents/structured.py',          'precise', 3, 'sim-greedy baseline (850)'),
    ('opponents/structured_tier900.py',  'precise', 3, 'sim-greedy strong (900)'),
    ('opponents/structured_tier1100.py', 'precise', 3, 'sim-greedy elite (1100)'),
    ('opponents/lb1224.py',              'precise', 3, 'sim-greedy tuned aggressive'),
    ('opponents/orbitbotnext.py',        'precise', 2, 'sim-greedy long horizon'),
    ('opponents/sundodge.py',            'precise', 2, 'sim-greedy + sun avoidance'),
    ('opponents/kaggle_low.py',          'precise', 1, 'parametric precise (800-level)'),

    # === MODERATE_PRESSURE (24% of field) ===
    ('opponents/proto1000.py',    'moderate', 3, 'proto1000 full framework'),
    ('opponents/reinforce958.py', 'moderate', 2, 'reinforce-heavy variant'),
    ('opponents/kaggle_mid.py',   'moderate', 2, 'parametric moderate'),
    ('opponents/tactical2026.py', 'moderate', 1, 'tactical sim-light'),

    # === HIGH_VOLUME (14% of field) ===
    ('opponents/kaggle_high.py',  'high_vol', 2, 'parametric high-vol'),
    ('opponents/kaggle_high2.py', 'high_vol', 1, 'hyper-spam variant'),
    ('opponents/smart_swarmer_v3.py', 'high_vol', 1, 'smart swarmer'),

    # === PASSIVE/DEFENSIVE (14% of field) ===
    ('opponents/peak1103.py',     'passive',  2, 'defensive structured'),
]

# Add BC-smart opponents (proto1000 backbone with tier-specific params)
BC_TIERS = ['800', '900', '1000+']

SEEDS = [42, 123, 777, 999, 314, 2718, 1618, 4242, 9999, 7777,
         1234, 5678, 3141, 2023, 8888][:N_SEEDS]


def run_game(agent_path, opp, seed):
    try:
        env = make('orbit_wars', configuration={'seed': seed}, debug=True)
        env.run([agent_path, opp])
        last = env.steps[-1]
        won = last[0].reward > last[1].reward
        return won, len(env.steps)
    except Exception as e:
        return None, 0


print(f"Arena bench: {AGENT}")
print(f"{len(POOL)} real bots + {len(BC_TIERS)} BC-smart tiers × {N_SEEDS} seeds")
print("=" * 80)

arch_w = {}
arch_g = {}

# Part 1: Real notebook bots
for opp_path, arch, weight, desc in POOL:
    opp_name = os.path.basename(opp_path).replace('.py', '')
    wins, games = 0, 0

    for seed in SEEDS:
        won, steps = run_game(AGENT, opp_path, seed)
        if won is not None:
            wins += int(won)
            games += 1

    wr = wins / games * 100 if games > 0 else 0
    print(f"  {opp_name:<22} [{arch:<8}] w={weight}: {wins}/{games} = {wr:.0f}%  ({desc})")

    arch_w[arch] = arch_w.get(arch, 0) + wins * weight
    arch_g[arch] = arch_g.get(arch, 0) + games * weight

# Part 2: BC-smart opponents
print(f"\n  --- BC-smart (proto1000 backbone, tier-parameterized) ---")
for tier in BC_TIERS:
    os.environ['BC_TIER'] = tier
    import opponents.bc_smart as bcs
    importlib.reload(bcs)

    wins, games = 0, 0
    for seed in SEEDS:
        won, steps = run_game(AGENT, 'opponents/bc_smart.py', seed)
        if won is not None:
            wins += int(won)
            games += 1

    wr = wins / games * 100 if games > 0 else 0
    weight = 2
    arch = 'bc_smart'
    print(f"  BC-smart-{tier:<13} [{arch:<8}] w={weight}: {wins}/{games} = {wr:.0f}%")
    arch_w[arch] = arch_w.get(arch, 0) + wins * weight
    arch_g[arch] = arch_g.get(arch, 0) + games * weight

# Summary
total_w = sum(arch_w.values())
total_g = sum(arch_g.values())

print(f"\n{'='*80}")
print(f"Weighted total: {total_w:.0f}/{total_g:.0f} = {total_w/total_g*100:.1f}%")
print(f"\nPer archetype:")
for arch in ['precise', 'moderate', 'high_vol', 'passive', 'bc_smart']:
    if arch in arch_g and arch_g[arch] > 0:
        print(f"  {arch:<10}: {arch_w[arch]:.0f}/{arch_g[arch]:.0f} = {arch_w[arch]/arch_g[arch]*100:.0f}%")
