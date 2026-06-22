"""
Build data/replay_index.jsonl: one line per replay with metadata.

Fields:
  episode_id, format ('2P'|'4P'), seed, steps, rewards, winner_idx,
  team_names, my_idx (slot where 'Borrun' is, or -1), my_won (bool),
  agents_meta (list of dicts), path (relative)

Usage:
  .venv/bin/python scripts/build_replay_index.py
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAYS = ROOT / 'replays'
OUT = ROOT / 'data' / 'replay_index.jsonl'


def me_slot(team_names):
    for i, t in enumerate(team_names):
        # User team identifiers in Kaggle
        if 'Borrun' in t or 'orbit-wars' in t.lower():
            return i
    return -1


def index_replay(path: Path):
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return {"path": str(path.relative_to(ROOT)), "error": str(e)}
    m = re.search(r'episode-(\d+)\.json', path.name)
    eid = int(m.group(1)) if m else None
    info = d.get('info', {})
    teams = info.get('TeamNames', [])
    agents = info.get('Agents', [])
    rewards = d.get('rewards', [])
    n = len(rewards)
    winner_idx = -1
    if rewards:
        max_r = max(rewards)
        if rewards.count(max_r) == 1:
            winner_idx = rewards.index(max_r)
    my_idx = me_slot(teams)
    my_won = (my_idx >= 0 and my_idx == winner_idx)
    return {
        "episode_id": eid,
        "format": f"{n}P",
        "seed": info.get('seed'),
        "steps": len(d.get('steps', [])),
        "rewards": rewards,
        "winner_idx": winner_idx,
        "team_names": teams,
        "my_idx": my_idx,
        "my_won": my_won,
        "agents_meta": agents,
        "path": str(path.relative_to(ROOT)),
    }


def main():
    records = []
    for sub in ['2p', '4p']:
        for p in sorted((REPLAYS / sub).glob('episode-*.json')):
            rec = index_replay(p)
            records.append(rec)
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    # Summary
    n2 = sum(1 for r in records if r.get('format') == '2P')
    n4 = sum(1 for r in records if r.get('format') == '4P')
    mine_2p = [r for r in records if r.get('format') == '2P' and r.get('my_idx', -1) >= 0]
    mine_4p = [r for r in records if r.get('format') == '4P' and r.get('my_idx', -1) >= 0]
    wins_2p = sum(1 for r in mine_2p if r.get('my_won'))
    wins_4p = sum(1 for r in mine_4p if r.get('my_won'))
    print(f"Indexed: {len(records)} replays  → {OUT}")
    print(f"  2P: {n2} total, {len(mine_2p)} with my team, "
          f"my WR = {wins_2p}/{len(mine_2p)} = "
          f"{100*wins_2p/max(1,len(mine_2p)):.1f}%")
    print(f"  4P: {n4} total, {len(mine_4p)} with my team, "
          f"my W = {wins_4p}/{len(mine_4p)} = "
          f"{100*wins_4p/max(1,len(mine_4p)):.1f}%")


if __name__ == '__main__':
    main()
