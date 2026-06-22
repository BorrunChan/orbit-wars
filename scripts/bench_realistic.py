"""
Realistic Kaggle bench — weighted opponent pool matching real field distribution.

Uses parameterized + real notebook bots to approximate the opponent mix at
each rating tier. Run with:
    .venv/bin/python scripts/bench_realistic.py agents/v132_situational.py [seeds]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kaggle_environments import make

AGENT = sys.argv[1] if len(sys.argv) > 1 else 'main.py'
N_SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

# Opponent pool weighted by real Kaggle 850-1000 distribution
# Each entry: (path, archetype, weight, approx_strength)
POOL = [
    # PRECISE/STRUCTURED (48% of field)
    ('opponents/structured.py',    'precise',  3, '850-950'),
    ('opponents/kaggle_low.py',    'precise',  3, '750-850'),
    ('opponents/lb1224.py',        'precise',  2, '900-1000'),
    ('opponents/orbitbotnext.py',  'precise',  2, '900-1000'),
    ('opponents/sundodge.py',      'precise',  1, '900-1000'),
    
    # MODERATE (24% of field)  
    ('opponents/proto1000.py',     'moderate', 2, '850-950'),
    ('opponents/kaggle_mid.py',    'moderate', 2, '800-900'),
    ('opponents/reinforce958.py',  'moderate', 1, '900-1000'),
    ('opponents/tactical2026.py',  'moderate', 1, '700-800'),
    
    # HIGH VOLUME (14% of field)
    ('opponents/kaggle_high.py',   'high_vol', 2, '800-900'),
    ('opponents/kaggle_high2.py',  'high_vol', 1, '750-850'),
    ('opponents/smart_swarmer_v3.py', 'high_vol', 1, '700-800'),
    
    # PASSIVE (14% of field)
    ('opponents/peak1103.py',      'passive',  1, '800-900'),
]

SEEDS = [42, 123, 777, 999, 314, 2718, 1618, 4242, 8675, 3090,
         1111, 2222, 3333, 4444, 5555][:N_SEEDS]

def run_game(agent_path, opp_path, seed):
    try:
        env = make('orbit_wars', configuration={'seed': seed}, debug=True)
        env.run([agent_path, opp_path])
        last = env.steps[-1]
        won = last[0].reward > last[1].reward
        steps = len(env.steps)
        return won, steps, None
    except Exception as e:
        return None, 0, str(e)

print(f"Bench: {AGENT} vs {len(POOL)} opponents × {N_SEEDS} seeds")
print("=" * 80)

results = {}
total_w, total_g = 0, 0
arch_results = {}

for opp_path, arch, weight, strength in POOL:
    opp_name = os.path.basename(opp_path).replace('.py', '')
    wins, games = 0, 0
    
    for seed in SEEDS:
        won, steps, err = run_game(AGENT, opp_path, seed)
        if err:
            print(f"  ERROR {opp_name} seed {seed}: {err}")
            continue
        if won is None:
            continue
        wins += int(won)
        games += 1
        status = 'W' if won else 'L'
    
    wr = wins / games * 100 if games > 0 else 0
    print(f"  {opp_name:<25} [{arch:<8}] {strength:<10}: {wins}/{games} = {wr:.0f}%")
    
    results[opp_name] = (wins, games, arch, weight)
    total_w += wins * weight
    total_g += games * weight
    
    if arch not in arch_results:
        arch_results[arch] = [0, 0]
    arch_results[arch][0] += wins * weight
    arch_results[arch][1] += games * weight

print(f"\n{'='*80}")
print(f"Weighted total: {total_w}/{total_g} = {total_w/total_g*100:.1f}%")
print(f"\nPer archetype:")
for arch in ['precise', 'moderate', 'high_vol', 'passive']:
    if arch in arch_results:
        w, g = arch_results[arch]
        print(f"  {arch:<10}: {w:.0f}/{g:.0f} = {w/g*100:.0f}%")
