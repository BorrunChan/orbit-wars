"""
Per-seed analyzer — runs main vs opponent across N seeds and prints per-game:
final scores (planets+fleets per player), ownership counts, total ships.

Usage:
    .venv/bin/python scripts/diagnose.py opponent [N]
"""
import logging, os, sys
from pathlib import Path
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet, Fleet

ROOT = Path(__file__).resolve().parent.parent
opp = sys.argv[1] if len(sys.argv) > 1 else "starter"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 15

agent_a = str(ROOT / "main.py")
agent_b = opp if opp in ("random", "starter") else str(ROOT / "agents" / f"{opp}.py")


def play(seed, swap=False):
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    p0, p1 = (agent_b, agent_a) if swap else (agent_a, agent_b)
    env.run([p0, p1])
    obs = env.steps[-1][0].observation
    planets = [Planet(*p) for p in obs["planets"]]
    fleets = [Fleet(*f) for f in obs["fleets"]]
    scores = [0, 0]; owned = [0, 0]
    for p in planets:
        if p.owner in (0, 1):
            scores[p.owner] += p.ships
            owned[p.owner] += 1
    for f in fleets:
        if f.owner in (0, 1):
            scores[f.owner] += f.ships
    me, opp_idx = (1, 0) if swap else (0, 1)
    r_me = env.steps[-1][me].reward
    return {
        "seed": seed, "swap": swap, "steps": len(env.steps),
        "me_score": scores[me], "opp_score": scores[opp_idx],
        "me_owned": owned[me], "opp_owned": owned[opp_idx],
        "reward": r_me,
    }


print(f"main vs {opp}, {N} seeds (each side):")
print(f"{'seed':>4} {'side':>4}  {'result':<5}  {'me_ships':>9} {'opp_ships':>9} {'me_pl':>5} {'opp_pl':>6}  steps")
losses = []
for s in range(N):
    for swap in (False, True):
        r = play(s, swap)
        side = "P1" if swap else "P0"
        res = "WIN" if r["reward"] > 0 else "LOSS" if r["reward"] < 0 else "DRAW"
        if res == "LOSS":
            losses.append(r)
        print(f"{r['seed']:>4} {side:>4}  {res:<5}  {r['me_score']:>9} {r['opp_score']:>9} {r['me_owned']:>5} {r['opp_owned']:>6}  {r['steps']}")

print(f"\nLosses: {len(losses)} / {2*N}")
if losses:
    print(f"Avg loss margin: {sum(l['opp_score']-l['me_score'] for l in losses)/len(losses):.1f} ships")
