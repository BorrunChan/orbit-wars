"""
Pre-generate initial Orbit Wars states for many seeds. Save as JSON for
remote RL training to consume without needing kaggle_environments.

For each seed:
  - Make env, run 1 turn (populates planets + assigns homes), save initial state

Output: data/initial_states_NP.json
   {"seed": N, "num_players": 2/4, "planets": [...], "fleets": [],
    "angular_velocity": 0.04, "initial_planets": [...]}
"""
import argparse, json, logging, sys
from pathlib import Path
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--format", choices=["2p", "4p"], default="4p")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    np_count = 2 if args.format == "2p" else 4
    out = args.out or str(ROOT / "data" / f"initial_states_{args.format}.json")

    print(f"Generating {args.n} initial states for {args.format}...")
    states = []
    for seed in range(args.n):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        # FIX: env needs to RUN with right num_players to trigger home assignment.
        # Without this, make() defaults to 2P layout (only players 0 and 1 get homes).
        # That's why earlier RL training had pos 2/3 always lose — they started with 0 planets!
        # Run just 1 step with random agents to assign homes, then capture step 0.
        env.reset(np_count)
        # Force 1 step to populate home assignments
        env.step([[] for _ in range(np_count)])
        obs = env.steps[0][0].observation
        s = {
            "seed": seed,
            "num_players": np_count,
            "step": obs.step,
            "angular_velocity": obs.angular_velocity,
            "planets": [list(p) for p in obs.planets],
            "fleets": [],
            "initial_planets": [list(p) for p in (obs.initial_planets or [])],
        }
        states.append(s)
        if (seed + 1) % 100 == 0:
            print(f"  {seed+1}/{args.n}", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(states, f)
    sz_mb = Path(out).stat().st_size / 1e6
    print(f"\nWrote {len(states)} states to {out} ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
