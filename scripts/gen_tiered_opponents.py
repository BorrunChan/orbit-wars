"""
Generate tiered variants of the structured framework (structured/lb1224/orbitbotnext).

Creates opponents at ~700, ~800, ~900, ~1000, ~1100 level by systematically
weakening or strengthening key decision parameters.

The structured framework has ~60 parameters. The ones that most affect strength:
  - SIM_HORIZON: how far the sim looks ahead (longer = smarter)
  - ATTACK margins: how many extra ships to send (higher = safer but slower)
  - VALUE mults: how much to value different target types
  - DEFENSE ratios: how aggressively to defend
  - REAR send ratios: how much to logistics from rear

"Weaker" opponents: shorter horizon, worse margins, less defense
"Stronger" opponents: longer horizon, tighter margins, more defense

Usage:
    .venv/bin/python scripts/gen_tiered_opponents.py
"""
import os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'opponents', 'structured.py')
OUT_DIR = os.path.join(ROOT, 'opponents')

# Read base
with open(BASE) as f:
    base_code = f.read()

# Parameter overrides per tier
# Format: {param_name: new_value}
TIERS = {
    'tier700': {
        'SIM_HORIZON': 50,
        'ROUTE_SEARCH_HORIZON': 25,
        'ATTACK_COST_TURN_WEIGHT': 0.7,   # overpays for attacks
        'HOSTILE_TARGET_VALUE_MULT': 1.4,  # less interested in enemies
        'NEUTRAL_MARGIN_BASE': 4,          # sends too many ships to neutrals
        'HOSTILE_MARGIN_BASE': 6,
        'PROACTIVE_DEFENSE_RATIO': 0.10,   # weak defense
        'REAR_SEND_RATIO_TWO_PLAYER': 0.3, # weak logistics
        'MULTI_SOURCE_TOP_K': 2,           # less coordination
        'BEHIND_DOMINATION': -0.10,
    },
    'tier900': {
        'SIM_HORIZON': 130,
        'ROUTE_SEARCH_HORIZON': 70,
        'ATTACK_COST_TURN_WEIGHT': 0.48,
        'HOSTILE_TARGET_VALUE_MULT': 2.0,
        'NEUTRAL_MARGIN_BASE': 1,
        'HOSTILE_MARGIN_BASE': 2,
        'PROACTIVE_DEFENSE_RATIO': 0.22,
        'REAR_SEND_RATIO_TWO_PLAYER': 0.70,
        'MULTI_SOURCE_TOP_K': 6,
        'BEHIND_DOMINATION': -0.25,
        'ELIMINATION_BONUS': 22.0,
    },
    'tier1100': {
        'SIM_HORIZON': 160,
        'ROUTE_SEARCH_HORIZON': 90,
        'ATTACK_COST_TURN_WEIGHT': 0.42,
        'SNIPE_COST_TURN_WEIGHT': 0.38,
        'HOSTILE_TARGET_VALUE_MULT': 2.2,
        'SAFE_NEUTRAL_VALUE_MULT': 1.4,
        'NEUTRAL_MARGIN_BASE': 1,
        'HOSTILE_MARGIN_BASE': 1,
        'HOSTILE_MARGIN_CAP': 8,
        'PROACTIVE_DEFENSE_RATIO': 0.28,
        'MULTI_ENEMY_PROACTIVE_RATIO': 0.30,
        'REAR_SEND_RATIO_TWO_PLAYER': 0.80,
        'REAR_SEND_RATIO_FOUR_PLAYER': 0.85,
        'MULTI_SOURCE_TOP_K': 8,
        'BEHIND_DOMINATION': -0.30,
        'AHEAD_DOMINATION': 0.25,
        'ELIMINATION_BONUS': 25.0,
        'REINFORCE_VALUE_MULT': 1.50,
    },
}

for tier_name, overrides in TIERS.items():
    code = base_code
    for param, value in overrides.items():
        pattern = rf'^({param}\s*=\s*).*$'
        if isinstance(value, float):
            replacement = rf'\g<1>{value}'
        elif isinstance(value, bool):
            replacement = rf'\g<1>{value}'
        else:
            replacement = rf'\g<1>{value}'
        code = re.sub(pattern, replacement, code, count=1, flags=re.MULTILINE)

    out_path = os.path.join(OUT_DIR, f'structured_{tier_name}.py')
    with open(out_path, 'w') as f:
        f.write(code)
    print(f"Generated {out_path} ({len(overrides)} params modified)")

print("Done!")
