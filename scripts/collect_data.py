"""
Run many games between agents, capture (state, action) pairs from WINNERS.

Idea: env.steps[i][player]['action'] already holds every agent's decision at
step i. We don't need to modify any opponent's code — env already exposes the
decision trace.

We run lots of games (mixed pairings, 2P and 4P), determine the winner of each
game, and save the winner's full trajectory of (obs, action) pairs. Later we
analyze these to extract behavior patterns.

Output:
    data/winner_traces.jsonl   — one line per (winner, game) trajectory

Usage:
    .venv/bin/python -u scripts/collect_data.py [num_seeds]
"""

import json, logging, os, sys, time, itertools
from pathlib import Path
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make

ROOT = Path(__file__).resolve().parent.parent
OPP_DIR = ROOT / "opponents"
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


def ref(name):
    if name in ("random", "starter"):
        return name
    if name == "main":
        return str(ROOT / "main.py")
    p = OPP_DIR / f"{name}.py"
    if p.exists():
        return str(p)
    p2 = ROOT / "agents" / f"{name}.py"
    if p2.exists():
        return str(p2)
    raise FileNotFoundError(name)


def summarize_state(obs, player):
    """Compact state summary for one player's POV."""
    def g(k, default=None):
        if isinstance(obs, dict):
            return obs.get(k, default)
        return getattr(obs, k, default)
    planets = g("planets", []) or []
    fleets = g("fleets", []) or []
    step = g("step", 0) or 0
    my_planets = []
    enemy_planets = []
    neutral_planets = []
    for p in planets:
        d = {"id": p[0], "x": round(p[2], 2), "y": round(p[3], 2),
             "ships": p[5], "prod": p[6]}
        if p[1] == player:
            my_planets.append(d)
        elif p[1] == -1:
            neutral_planets.append(d)
        else:
            d["owner"] = p[1]
            enemy_planets.append(d)
    my_fleets = []
    enemy_fleets = []
    for f in fleets:
        d = {"x": round(f[2], 2), "y": round(f[3], 2),
             "ang": round(f[4], 3), "ships": f[6]}
        if f[1] == player:
            my_fleets.append(d)
        else:
            d["owner"] = f[1]
            enemy_fleets.append(d)
    return {
        "step": step,
        "my_planets": my_planets,
        "enemy_planets": enemy_planets,
        "neutral_planets": neutral_planets,
        "my_fleets": my_fleets,
        "enemy_fleets": enemy_fleets,
    }


def run_game(agents, seed, game_meta):
    """Run one game, return list of trajectories per winner."""
    env = make("orbit_wars", configuration={"seed": seed}, debug=False)
    try:
        env.run([ref(a) for a in agents])
    except Exception as e:
        return None, str(e)
    final = env.steps[-1]
    rewards = [s.reward for s in final]
    # Determine winner(s) — at most one usually
    max_r = max(rewards)
    winners = [i for i, r in enumerate(rewards) if r == max_r and r > 0]
    if not winners:
        return [], None  # no winner (draw/timeout)

    trajectories = []
    for w in winners:
        traj = []
        for step_idx, step_data in enumerate(env.steps):
            if step_idx >= len(env.steps) - 1:
                continue
            entry = step_data[w]
            obs = entry.observation
            action = entry.action
            if action is None:
                action = []
            state = summarize_state(obs, w)
            traj.append({"state": state, "action": action})
        trajectories.append({
            "winner_agent": agents[w],
            "winner_pos": w,
            "num_players": len(agents),
            "opponents": [agents[i] for i in range(len(agents)) if i != w],
            "seed": seed,
            "steps": len(env.steps),
            "traj": traj,
            "meta": game_meta,
        })
    return trajectories, None


def make_2p_pairings(agents):
    out = []
    for i in range(len(agents)):
        for j in range(len(agents)):
            if i == j:
                continue
            out.append((agents[i], agents[j]))
    return out


def make_4p_lineups(agents):
    """Generate diverse 4P lineups (sampled)."""
    out = []
    # All-same
    for a in agents:
        out.append([a, a, a, a])
    # All-different (one of each, rotate)
    if len(agents) >= 4:
        out.append(agents[:4])
    # Two of A + two of B
    for a, b in itertools.combinations(agents, 2):
        out.append([a, a, b, b])
        out.append([a, b, a, b])
    return out


def main():
    num_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    pool = ["lb1224", "proto1000", "structured", "main"]

    out_file = DATA_DIR / "winner_traces.jsonl"
    if out_file.exists():
        out_file.unlink()  # fresh start

    pairings_2p = make_2p_pairings(pool)
    lineups_4p = make_4p_lineups(pool)

    total_games = num_seeds * (len(pairings_2p) + len(lineups_4p))
    print(f"Plan: {len(pairings_2p)} 2P pairings + {len(lineups_4p)} 4P lineups "
          f"× {num_seeds} seeds = {total_games} games")
    t0 = time.time()
    saved = 0
    errors = 0
    skipped_draws = 0
    f_out = open(out_file, "a")
    for seed in range(num_seeds):
        # 2P
        for (a, b) in pairings_2p:
            meta = {"mode": "2p", "pair": [a, b]}
            trajs, err = run_game([a, b], seed, meta)
            if err:
                errors += 1
                continue
            if not trajs:
                skipped_draws += 1
                continue
            for t in trajs:
                f_out.write(json.dumps(t) + "\n")
                f_out.flush()
                saved += 1
        # 4P
        for lineup in lineups_4p:
            meta = {"mode": "4p", "lineup": lineup}
            trajs, err = run_game(lineup, seed, meta)
            if err:
                errors += 1
                continue
            if not trajs:
                skipped_draws += 1
                continue
            for t in trajs:
                f_out.write(json.dumps(t) + "\n")
                f_out.flush()
                saved += 1
        print(f"[seed {seed+1}/{num_seeds}] saved={saved}  errors={errors}  "
              f"draws={skipped_draws}  t={time.time()-t0:.0f}s", flush=True)
    f_out.close()
    print(f"\nDone. {saved} winner trajectories saved to {out_file}")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
