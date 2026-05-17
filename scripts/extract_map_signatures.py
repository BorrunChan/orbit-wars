"""
Sample many seeds, extract per-map geometric signature.

Map space is small (only angular_velocity + planet_count groups + positions),
so 1500 seeds should saturate the distribution. Output one row per seed.

Usage:
  .venv/bin/python scripts/extract_map_signatures.py [n_seeds=1500]
"""
import argparse, json, logging, math, sys, time
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "map_signatures.jsonl"

BOARD = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0


def extract_signature(obs, seed):
    """Compute geometric features of the static map."""
    planets = obs.planets
    av = obs.angular_velocity
    n_p = len(planets)

    # Per-planet polar coords from board center
    radii = [math.hypot(p[2] - CENTER, p[3] - CENTER) for p in planets]
    # Owner mix at step 0: -1 = neutral; 0..N-1 = player
    owners = [p[1] for p in planets]
    n_neutral = sum(1 for o in owners if o == -1)
    n_owned = n_p - n_neutral

    # Production / fleet endowment
    prods = [p[6] for p in planets]
    ships = [p[5] for p in planets]
    radii_planet = [p[4] for p in planets]  # planet body radius (not orbital)

    # Min pairwise planet distance (proxy for crowding)
    min_dist = float("inf")
    for i in range(n_p):
        for j in range(i + 1, n_p):
            d = math.hypot(planets[i][2] - planets[j][2],
                           planets[i][3] - planets[j][3])
            if d < min_dist:
                min_dist = d

    # Rotating vs static planets: orbital_r + r < ROTATION_RADIUS_LIMIT(=50) means rotating
    n_rotating = sum(1 for p, r in zip(planets, radii)
                      if r + p[4] < 50.0)
    n_static = n_p - n_rotating

    # Inner shell (close to sun) vs outer
    n_inner = sum(1 for r in radii if r < 25)   # close half
    n_outer = sum(1 for r in radii if r >= 25)

    # Sun-blocking density: how many planet pairs have sun roughly between them
    # (crude: orbital_r on opposite sides + close to perpendicular through sun)
    n_sun_pair = 0
    for i in range(n_p):
        for j in range(i + 1, n_p):
            xi, yi = planets[i][2] - CENTER, planets[i][3] - CENTER
            xj, yj = planets[j][2] - CENTER, planets[j][3] - CENTER
            # angle between vectors from center
            dot = xi*xj + yi*yj
            mi = math.hypot(xi, yi); mj = math.hypot(xj, yj)
            if mi*mj < 1e-9:
                continue
            cos_a = dot / (mi*mj)
            # both close to center + angle near 180 → sun in between
            if cos_a < -0.85 and mi < 35 and mj < 35:
                n_sun_pair += 1

    avg_r = sum(radii) / n_p
    std_r = math.sqrt(sum((r - avg_r)**2 for r in radii) / n_p)

    return {
        "seed": seed,
        "angular_velocity": round(av, 5),
        "n_planets": n_p,
        "n_planet_groups": n_p // 4,
        "n_rotating": n_rotating,
        "n_static": n_static,
        "rotating_frac": round(n_rotating / n_p, 3),
        "min_planet_dist": round(min_dist, 2),
        "avg_orbital_radius": round(avg_r, 2),
        "std_orbital_radius": round(std_r, 2),
        "n_inner_planets": n_inner,
        "n_outer_planets": n_outer,
        "n_sun_block_pairs": n_sun_pair,
        "total_prod": sum(prods),
        "avg_prod": round(sum(prods) / n_p, 2),
        "avg_planet_radius": round(sum(radii_planet) / n_p, 3),
        "avg_initial_ships_per_planet": round(sum(ships) / n_p, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = 0
    with open(OUT, "w") as f:
        for seed in range(args.n):
            # make() alone populates step 0 (planets + angular_velocity).
            # No need to actually play the game.
            env = make("orbit_wars", configuration={"seed": seed}, debug=False)
            s0 = env.steps[0][0].observation
            sig = extract_signature(s0, seed)
            f.write(json.dumps(sig) + "\n")
            n_ok += 1
            if (seed + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed * (args.n - seed - 1) / max(1, seed + 1)
                print(f"[{seed+1}/{args.n}] elapsed={elapsed:.0f}s eta={eta:.0f}s",
                      flush=True)
    print(f"\nWrote {n_ok} signatures to {OUT} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
