"""
Model real Kaggle opponent styles + mainstream playstyle from the full
replay corpus, joined with live leaderboard rating.

For every non-Borrun slot in every replay, extract per-game style features,
aggregate per opponent (TeamName), then join the team's current leaderboard
Score so we can correlate STYLE -> RATING (i.e. which styles actually win
across the whole 2994-team field, not just vs us).

Outputs:
  data/opponent_styles.jsonl   per-opponent aggregated style + rating
  data/opponent_styles.md      human-readable taxonomy + meta findings

Usage:
  .venv/bin/python scripts/model_opponents.py
"""
import csv
import glob
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPLAY_DIRS = [ROOT / 'replays' / '2p', ROOT / 'replays' / '4p']
LB_GLOB = '/tmp/lb_extract/*publicleaderboard*.csv'
OUT_JSONL = ROOT / 'data' / 'opponent_styles.jsonl'
OUT_MD = ROOT / 'data' / 'opponent_styles.md'


def load_leaderboard():
    name_to_score = {}
    name_to_rank = {}
    files = glob.glob(LB_GLOB)
    if not files:
        return name_to_score, name_to_rank
    with open(files[0], encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                name_to_score[r['TeamName']] = float(r['Score'])
                name_to_rank[r['TeamName']] = int(r['Rank'])
            except (ValueError, KeyError):
                pass
    return name_to_score, name_to_rank


def iter_replays():
    seen = set()
    for d in REPLAY_DIRS:
        if not d.exists():
            continue
        for p in d.glob('*.json'):
            # dedupe by episode id embedded in filename
            stem = p.stem.replace('episode-', '').replace('-replay', '')
            if stem in seen:
                continue
            seen.add(stem)
            yield p


def game_features_for_slot(d, slot, n_players):
    """Per-game style features for the agent at `slot`."""
    steps = d['steps']
    n_steps = max(1, len(steps) - 1)
    raw_launches = 0
    fleet_sizes = []
    first_launch = None
    planets_at = {}
    milestones = [25, 50, 100, 150]
    peak_planets = 0
    for si, step in enumerate(steps[:-1]):
        if slot >= len(step):
            continue
        s = step[slot]
        act = s.get('action') or []
        for a in act:
            if isinstance(a, list) and len(a) >= 3:
                raw_launches += 1
                fleet_sizes.append(a[2])
                if first_launch is None:
                    first_launch = si
        # planet ownership via slot 0's observation (global, full-info)
        obs = (step[0].get('observation') if step and step[0] else None) or {}
        pls = obs.get('planets', [])
        if pls:
            owned = sum(1 for p in pls if p[1] == slot)
            peak_planets = max(peak_planets, owned)
            if si in milestones:
                planets_at[si] = owned
    return {
        'n_steps': n_steps,
        'launch_rate': raw_launches / n_steps,
        'avg_fleet': st.mean(fleet_sizes) if fleet_sizes else 0,
        'med_fleet': st.median(fleet_sizes) if fleet_sizes else 0,
        'first_launch': first_launch if first_launch is not None else n_steps,
        'planets_25': planets_at.get(25, 0),
        'planets_50': planets_at.get(50, 0),
        'planets_100': planets_at.get(100, 0),
        'peak_planets': peak_planets,
    }


def archetype(launch_rate, avg_fleet):
    if launch_rate >= 3.0:
        return 'high_volume_swarm'
    if launch_rate >= 1.2:
        return 'moderate_pressure'
    if launch_rate >= 0.3:
        return 'precise_structured'
    return 'passive_turtle'


def main():
    name_to_score, name_to_rank = load_leaderboard()
    print(f"leaderboard entries: {len(name_to_score)}")

    agg = defaultdict(lambda: {
        'games': 0, 'wins': 0,
        'launch_rates': [], 'fleets': [], 'first_launches': [],
        'p25': [], 'p50': [], 'p100': [], 'peak': [],
        'formats': defaultdict(int),
    })

    n_replays = 0
    for p in iter_replays():
        try:
            d = json.load(open(p))
        except Exception:
            continue
        teams = d.get('info', {}).get('TeamNames', [])
        rew = d.get('rewards', [])
        n = len(rew)
        if not teams or n == 0:
            continue
        n_replays += 1
        max_r = max(rew) if rew else None
        for slot, name in enumerate(teams):
            if name == 'Borrun':
                continue
            won = (rew[slot] == max_r and rew.count(max_r) == 1)
            feat = game_features_for_slot(d, slot, n)
            a = agg[name]
            a['games'] += 1
            a['wins'] += int(won)
            a['launch_rates'].append(feat['launch_rate'])
            if feat['avg_fleet'] > 0:
                a['fleets'].append(feat['avg_fleet'])
            a['first_launches'].append(feat['first_launch'])
            a['p25'].append(feat['planets_25'])
            a['p50'].append(feat['planets_50'])
            a['p100'].append(feat['planets_100'])
            a['peak'].append(feat['peak_planets'])
            a['formats'][f'{n}P'] += 1

    print(f"replays processed: {n_replays}")

    records = []
    for name, a in agg.items():
        g = a['games']
        lr = st.mean(a['launch_rates']) if a['launch_rates'] else 0
        fl = st.mean(a['fleets']) if a['fleets'] else 0
        rec = {
            'opp_name': name,
            'games': g,
            'wins_vs_field': a['wins'],
            'winrate_in_our_games': round(100 * a['wins'] / g, 1),
            'launch_rate': round(lr, 2),
            'avg_fleet': round(fl, 1),
            'med_first_launch': sorted(a['first_launches'])[len(a['first_launches']) // 2],
            'planets_25': round(st.mean(a['p25']), 1),
            'planets_50': round(st.mean(a['p50']), 1),
            'planets_100': round(st.mean(a['p100']), 1),
            'peak_planets': round(st.mean(a['peak']), 1),
            'formats': dict(a['formats']),
            'archetype': archetype(lr, fl),
            'lb_score': name_to_score.get(name),
            'lb_rank': name_to_rank.get(name),
        }
        records.append(rec)

    records.sort(key=lambda r: (r['lb_score'] is None, -(r['lb_score'] or 0)))

    OUT_JSONL.parent.mkdir(exist_ok=True)
    with open(OUT_JSONL, 'w') as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"wrote {len(records)} opponents -> {OUT_JSONL}")

    # ---- META ANALYSIS ----
    rated = [r for r in records if r['lb_score'] is not None and r['games'] >= 1]
    md = ["# Kaggle Opponent Styles + Mainstream Playstyle (data-driven)",
          "",
          f"Built from {n_replays} replays, {len(records)} unique opponents, "
          f"{len(rated)} with a live leaderboard score.",
          ""]

    # Archetype distribution + mean rating
    md.append("## Archetype distribution (weighted by leaderboard rating)")
    md.append("")
    md.append("| archetype | n_opps | mean_lb_score | median_lb_score | mean_launch_rate |")
    md.append("|---|---:|---:|---:|---:|")
    by_arch = defaultdict(list)
    for r in rated:
        by_arch[r['archetype']].append(r)
    for arch in ['high_volume_swarm', 'moderate_pressure',
                 'precise_structured', 'passive_turtle']:
        rs = by_arch.get(arch, [])
        if not rs:
            md.append(f"| {arch} | 0 | - | - | - |")
            continue
        scores = [r['lb_score'] for r in rs]
        lrs = [r['launch_rate'] for r in rs]
        md.append(f"| {arch} | {len(rs)} | {st.mean(scores):.0f} | "
                  f"{st.median(scores):.0f} | {st.mean(lrs):.2f} |")
    md.append("")

    # Correlation: launch_rate vs lb_score, expansion vs lb_score
    def corr(xs, ys):
        if len(xs) < 3:
            return 0.0
        mx, my = st.mean(xs), st.mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
        dy = math.sqrt(sum((y - my) ** 2 for y in ys))
        return num / (dx * dy) if dx * dy else 0.0

    md.append("## What correlates with high leaderboard rating?")
    md.append("")
    feats = [('launch_rate', 'launch_rate'),
             ('avg_fleet', 'avg_fleet'),
             ('planets_50', 'planets@50'),
             ('planets_100', 'planets@100'),
             ('peak_planets', 'peak_planets'),
             ('med_first_launch', 'first_launch_step')]
    md.append("| feature | Pearson r vs lb_score |")
    md.append("|---|---:|")
    for key, label in feats:
        xs = [r[key] for r in rated]
        ys = [r['lb_score'] for r in rated]
        md.append(f"| {label} | {corr(xs, ys):+.3f} |")
    md.append("")

    # Top-30 rated opponents and their styles
    md.append("## Top-30 rated opponents we've faced — their styles")
    md.append("")
    md.append("| rank | opp | lb_score | launch/t | fleet | p@50 | p@100 | peak | archetype |")
    md.append("|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rated[:30]:
        md.append(f"| {r['lb_rank']} | {r['opp_name'][:22]} | {r['lb_score']:.0f} | "
                  f"{r['launch_rate']} | {r['avg_fleet']} | {r['planets_50']} | "
                  f"{r['planets_100']} | {r['peak_planets']} | {r['archetype']} |")
    md.append("")

    # High vs low rating cohort comparison
    rated_sorted = sorted(rated, key=lambda r: r['lb_score'], reverse=True)
    top = rated_sorted[:max(5, len(rated_sorted) // 4)]
    bot = rated_sorted[-max(5, len(rated_sorted) // 4):]

    def cohort(rs, label):
        return (f"| {label} | {st.mean([r['lb_score'] for r in rs]):.0f} | "
                f"{st.mean([r['launch_rate'] for r in rs]):.2f} | "
                f"{st.mean([r['avg_fleet'] for r in rs]):.1f} | "
                f"{st.mean([r['planets_50'] for r in rs]):.1f} | "
                f"{st.mean([r['planets_100'] for r in rs]):.1f} | "
                f"{st.mean([r['peak_planets'] for r in rs]):.1f} |")
    md.append("## Top-quartile vs bottom-quartile rated opponents")
    md.append("")
    md.append("| cohort | mean_lb | launch/t | fleet | p@50 | p@100 | peak |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    md.append(cohort(top, 'TOP 25%'))
    md.append(cohort(bot, 'BOTTOM 25%'))
    md.append("")

    OUT_MD.write_text("\n".join(md))
    print(f"wrote meta report -> {OUT_MD}")


if __name__ == '__main__':
    main()
