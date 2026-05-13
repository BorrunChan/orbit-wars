"""
Run main.py vs an opponent across multiple seeds and report results.

Usage:
    .venv/bin/python scripts/run_match.py                  # vs random, 5 seeds
    .venv/bin/python scripts/run_match.py random 10        # vs random, 10 seeds
    .venv/bin/python scripts/run_match.py main.py 3        # mirror match
    .venv/bin/python scripts/run_match.py random 1 --html  # save HTML replay
"""

import logging
import sys
from pathlib import Path

logging.getLogger("kaggle_environments").setLevel(logging.ERROR)

from kaggle_environments import make  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run(opponent: str, num_seeds: int, save_html: bool) -> None:
    wins = losses = draws = 0
    for seed in range(num_seeds):
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run([str(ROOT / "main.py"), opponent])
        r0, r1 = env.steps[-1][0].reward, env.steps[-1][1].reward
        steps = len(env.steps)
        if r0 > r1:
            outcome, color = "WIN ", "\033[32m"
            wins += 1
        elif r0 < r1:
            outcome, color = "LOSS", "\033[31m"
            losses += 1
        else:
            outcome, color = "DRAW", "\033[33m"
            draws += 1
        print(f"  seed={seed:>3}  {color}{outcome}\033[0m  reward=({r0}, {r1})  steps={steps}")

        if save_html and seed == 0:
            html_path = ROOT / "replay.html"
            html_path.write_text(env.render(mode="html"))
            print(f"  -> replay saved to {html_path}")

    print(f"\nvs {opponent}: {wins}W / {losses}L / {draws}D out of {num_seeds}")


if __name__ == "__main__":
    opponent = sys.argv[1] if len(sys.argv) > 1 else "random"
    num_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    save_html = "--html" in sys.argv
    run(opponent, num_seeds, save_html)
