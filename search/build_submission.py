"""Bundle a search candidate into submission.tar.gz.
Archive root: main.py(=agent_main) + params.json + orbit_lite/.

    python search/build_submission.py --params search/_cand/<tag>/params.json --out search/sub_<tag>.tar.gz
    # or pass an inline params file you wrote yourself.
"""
from __future__ import annotations
import argparse, os, shutil, tarfile, pathlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEARCH = os.path.join(ROOT, "search")
ORBIT = os.path.join(ROOT, "single-size", "orbit_lite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build = os.path.join(SEARCH, "_build")
    if os.path.exists(build):
        shutil.rmtree(build)
    os.makedirs(build)
    shutil.copy2(os.path.join(SEARCH, "agent_main.py"), os.path.join(build, "main.py"))
    shutil.copy2(os.path.join(ROOT, args.params) if not os.path.isabs(args.params) else args.params,
                 os.path.join(build, "params.json"))
    shutil.copytree(ORBIT, os.path.join(build, "orbit_lite"),
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    out = os.path.join(ROOT, args.out) if not os.path.isabs(args.out) else args.out
    with tarfile.open(out, "w:gz") as tar:
        for p in sorted(pathlib.Path(build).rglob("*")):
            tar.add(p, arcname=p.relative_to(build), recursive=False)
    shutil.rmtree(build)
    print(f"built {out} ({os.path.getsize(out)} bytes)")


if __name__ == "__main__":
    main()
