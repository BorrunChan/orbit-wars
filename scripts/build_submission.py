"""
Build submission.tar.gz from current main.py + dependencies.

Usage:
  .venv/bin/python scripts/build_submission.py
"""
import os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

files = ["main.py", "sim.py", "shot_model.py", "value_4p_model.py"]
for f in files:
    if not (ROOT / f).exists():
        print(f"MISSING: {f}")
        sys.exit(1)

# Smoke test: import + run 1 game
import importlib.util
spec = importlib.util.spec_from_file_location("main", "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(f"main.py loaded OK: {m.__doc__[:80] if m.__doc__ else 'no docstring'}...")

import logging
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
from kaggle_environments import make
env = make("orbit_wars", configuration={"seed": 7}, debug=False)
env.run([str(ROOT / "main.py"), "random"])
print(f"2P smoke: rewards={[s.reward for s in env.steps[-1]]}, steps={len(env.steps)}")
env = make("orbit_wars", configuration={"seed": 7}, debug=False)
env.run([str(ROOT / "main.py"), "random", "random", "random"])
print(f"4P smoke: rewards={[s.reward for s in env.steps[-1]]}, steps={len(env.steps)}")

print()
print("Packaging...")
# COPYFILE_DISABLE=1 prevents macOS from adding ._ resource fork files
env = {**os.environ, "COPYFILE_DISABLE": "1"}
subprocess.check_call(["tar", "czf", "submission.tar.gz"] + files, env=env)
size = (ROOT / "submission.tar.gz").stat().st_size
print(f"submission.tar.gz: {size/1024:.0f} KB")
subprocess.check_call(["tar", "tzvf", "submission.tar.gz"])
print("\n✅ submission.tar.gz ready")
