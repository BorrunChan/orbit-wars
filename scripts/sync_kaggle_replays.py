"""
One-command Kaggle replay sync + analysis rebuild.

Pulls episodes for our submissions (default: all; or --submission <id>),
downloads any replays not already in replays/, classifies 2P/4P, then rebuilds
the index + opponent DB + style model. Closes the deploy->measure loop so we
can validate (on real Kaggle games) hypotheses the anti-correlated local bench
cannot — e.g. whether v130's logistics layer actually lifts us past ~800.

Usage:
  .venv/bin/python scripts/sync_kaggle_replays.py                 # all submissions
  .venv/bin/python scripts/sync_kaggle_replays.py --submission 52844500   # one
  .venv/bin/python scripts/sync_kaggle_replays.py --latest        # newest only
"""
import argparse
import csv
import io
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = str(ROOT / '.venv' / 'bin' / 'kaggle')
COMP = 'orbit-wars'
REPLAYS = ROOT / 'replays'


def run(args):
    return subprocess.run([KAGGLE] + args, capture_output=True, text=True)


def get_submission_ids(mode):
    r = run(['competitions', 'submissions', COMP, '--csv'])
    ids = []
    for row in csv.DictReader(io.StringIO(r.stdout)):
        ref = row.get('ref', '').strip()
        if ref.isdigit():
            ids.append(ref)
    if mode == 'latest':
        return ids[:1]
    return ids


def existing_episode_ids():
    s = set()
    for sub in ('2p', '4p'):
        for p in (REPLAYS / sub).glob('episode-*.json'):
            s.add(p.stem.replace('episode-', ''))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--submission', help='single submission id')
    ap.add_argument('--latest', action='store_true', help='newest submission only')
    ap.add_argument('--no-rebuild', action='store_true')
    args = ap.parse_args()

    (REPLAYS / '2p').mkdir(parents=True, exist_ok=True)
    (REPLAYS / '4p').mkdir(parents=True, exist_ok=True)

    if args.submission:
        subs = [args.submission]
    else:
        subs = get_submission_ids('latest' if args.latest else 'all')
    print(f"submissions to scan: {len(subs)}")

    # collect all episode ids
    all_eps = set()
    for s in subs:
        r = run(['competitions', 'episodes', s, '--csv'])
        for line in r.stdout.splitlines()[1:]:
            eid = line.split(',')[0].strip()
            if eid.isdigit():
                all_eps.add(eid)
    have = existing_episode_ids()
    missing = sorted(all_eps - have)
    print(f"episodes: {len(all_eps)} total, {len(have)} local, {len(missing)} to download")

    # download missing into a staging dir, then classify
    stage = Path('/tmp/kaggle_sync')
    stage.mkdir(exist_ok=True)
    ok = fail = 0
    t0 = time.time()
    for i, e in enumerate(missing):
        r = run(['competitions', 'replay', e, '-p', str(stage), '-q'])
        if r.returncode == 0:
            ok += 1
        else:
            fail += 1
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(missing)} ok={ok} fail={fail} {time.time()-t0:.0f}s",
                  flush=True)
    print(f"downloaded: {ok} ok, {fail} fail")

    # classify staged files into replays/2p|4p
    moved = 0
    for p in stage.glob('*.json'):
        m = re.search(r'(\d{6,})', p.name)
        if not m:
            continue
        eid = m.group(1)
        if eid in have:
            continue
        try:
            d = json.load(open(p))
            n = len(d.get('rewards', []))
        except Exception:
            continue
        if n not in (2, 4):
            continue
        dest = REPLAYS / ('2p' if n == 2 else '4p') / f'episode-{eid}.json'
        p.replace(dest)
        have.add(eid)
        moved += 1
    print(f"archived into replays/: {moved}")

    if not args.no_rebuild and moved:
        print("\nrebuilding analysis...")
        for script in ('build_replay_index.py', 'build_opponents_db.py',
                       'model_opponents.py'):
            subprocess.run([str(ROOT / '.venv' / 'bin' / 'python'),
                            str(ROOT / 'scripts' / script)])
    print("\ndone. To see v130's real per-archetype performance, inspect "
          "data/replay_index.jsonl (filter description/episode by date) and "
          "data/opponent_styles.md.")


if __name__ == '__main__':
    main()
