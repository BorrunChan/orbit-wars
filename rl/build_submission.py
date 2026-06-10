"""Bundle the value-agent into submission.tar.gz.

Archive root: main.py(=value_agent) + main_exp48.py + features.py + value_net.py
+ value.json (weights) + orbit_lite/.

    python rl/build_submission.py --weights rl/data/value_latest.json --out rl/submission_rl.tar.gz
"""
from __future__ import annotations
import argparse, os, shutil, tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out", default="rl/submission_rl.tar.gz")
    args = ap.parse_args()

    build = os.path.join(ROOT, "rl", "_build")
    if os.path.exists(build):
        shutil.rmtree(build)
    os.makedirs(build)

    shutil.copy2(os.path.join(ROOT, "rl", "value_agent.py"), os.path.join(build, "main.py"))
    shutil.copy2(os.path.join(ROOT, "single-size", "main_exp48.py"), os.path.join(build, "main_exp48.py"))
    shutil.copy2(os.path.join(ROOT, "rl", "features.py"), os.path.join(build, "features.py"))
    shutil.copy2(os.path.join(ROOT, "rl", "value_net.py"), os.path.join(build, "value_net.py"))
    shutil.copy2(os.path.join(ROOT, args.weights), os.path.join(build, "value.json"))
    shutil.copytree(os.path.join(ROOT, "single-size", "orbit_lite"),
                    os.path.join(build, "orbit_lite"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    out = os.path.join(ROOT, args.out)
    with tarfile.open(out, "w:gz") as tar:
        for p in sorted(__import__("pathlib").Path(build).rglob("*")):
            tar.add(p, arcname=p.relative_to(build), recursive=False)
    shutil.rmtree(build)
    print(f"built {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
