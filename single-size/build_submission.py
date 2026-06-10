#!/usr/bin/env python
"""Build submission.tar.gz for the Producer Hybrid v4 (single-size) agent.

Archive layout (root):
    main.py
    orbit_lite/...

Mirrors the notebook's build cell, run locally from this subproject.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default="main.py",
                    help="source file (in single-size/) to ship AS main.py")
    ap.add_argument("--out", default="submission.tar.gz")
    args = ap.parse_args()

    src_main = HERE / args.main
    if not src_main.exists():
        raise FileNotFoundError(f"main source not found: {src_main}")

    build_dir = HERE / "build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src_main, build_dir / "main.py")

    orbit_lite_src = HERE / "orbit_lite"
    if not orbit_lite_src.exists():
        raise FileNotFoundError(f"orbit_lite not found at {orbit_lite_src}")
    shutil.copytree(
        orbit_lite_src,
        build_dir / "orbit_lite",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )

    out = HERE / args.out
    with tarfile.open(out, "w:gz") as tar:
        for path in sorted(build_dir.rglob("*")):
            tar.add(path, arcname=path.relative_to(build_dir), recursive=False)
    shutil.rmtree(build_dir)
    print(f"Built {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
