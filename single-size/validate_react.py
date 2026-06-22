"""V115 gate for direction (4): does injecting opponent reactions into the
producer's projection change the RELATIVE ordering of MY candidate launches?

If only absolute scores shift (ranking unchanged) -> it's v115, abort.
If the ranking / argmax / roi-crossing changes -> reactive opp model is viable.

Reuses orbit_lite primitives to replicate plan_lite_waves' candidate generation
and scoring, scores under (A) do-nothing baseline and (B) opponents-also-expand,
on real mid-game 4P frames captured from a local game.
"""
from __future__ import annotations
import logging, os, sys
logging.getLogger("kaggle_environments").setLevel(logging.ERROR)
os.environ["KAGGLE_ENVIRONMENTS_QUIET"] = "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from kaggle_environments import make

from orbit_lite.obs import parse_obs
from orbit_lite.movement import MovementConfig
from orbit_lite.movement_step import (
    LaunchEntries, ensure_planet_movement, infer_planned_launches_from_entries,
    apply_private_planned_launches,
)
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices, build_target_shortlist, capture_floor, safe_drain,
    reachable_mask, make_launch_set, score_candidates,
)
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.geometry import fleet_speed

HOR = 13  # 4P horizon


def mcfg(pc):
    return MovementConfig(movement_horizon=HOR, drift_epsilon=1e-3,
                          track_fleets=True, player_count=int(pc), max_tracked_fleets=128)


def gen_and_score(obs_tensors, *, react):
    """Return (cand_target_planet_ids[list], scores[tensor]) for my candidates,
    optionally after injecting opponent expansion reactions."""
    obs = parse_obs(obs_tensors)
    pid = int(obs.player_id); P = obs.P
    dev = obs.device; dt = obs.ships.dtype
    pc = 4
    mov = ensure_planet_movement(obs_tensors=obs_tensors, expected_cfg=mcfg(pc), cached_movement=None)

    if react:
        _inject_opponent_expansion(mov, obs, obs_tensors, pid, pc)

    cache = build_distance_cache(mov, max_k=HOR)
    status = mov.garrison_status(max_horizon=HOR)
    prod = mov.planet_prod
    alive_by_step = mov.alive_by_step[:HOR + 1]

    K_eta = max(1, min(HOR, HOR))
    source_mask = obs.owned & obs.alive & (obs.ships >= 5.0)
    if not bool(source_mask.any()):
        return [], torch.zeros(0)
    source_idx, source_exists = _candidate_indices(obs.ships, source_mask, min(6, P))
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, status, cache,
        config=type("C", (), {"max_offensive_targets": 7, "max_defensive_targets": 2})(),
        K_eta=K_eta, H=HOR, prod=prod, source_mask=source_mask)
    if not bool(target_exists.any()):
        return [], torch.zeros(0)
    S = int(source_idx.shape[0]); T = int(target_idx.shape[0])
    src_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dt)
    H_eff = torch.full((), float(HOR), dtype=dt, device=dev)
    drain = safe_drain(status, source_idx=source_idx, source_ships=src_ships, H_eff=H_eff, player_id=pid)
    eta_cap = torch.full((T,), float(K_eta), dtype=dt, device=dev)
    floor = capture_floor(status, target_idx=target_idx, k_max=K_eta, capture_overhead=1.0, player_id=pid)
    K = int(floor.shape[-1])
    sizes = drain.view(S, 1).expand(S, T).floor()
    active = reachable_mask(mov, source_idx=source_idx, target_idx=target_idx,
                            fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap).squeeze(-1)
    aim = intercept_angle(mov, source_idx.unsqueeze(1), target_idx.unsqueeze(0), sizes, active=active)
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))
    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at = torch.ones(S, T, dtype=dt, device=dev)
    src_neq = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (viable & (sizes >= floor_at) & (sizes >= 1.0) & src_neq
             & source_exists.view(S, 1) & target_exists.view(1, T))
    L = 1; C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    launches = make_launch_set(source_slots=cand_src, target_slots=cand_tgt.unsqueeze(-1).expand(C, L),
                               ships=cand_send, eta=cand_eta,
                               valid=cand_active & cand_valid.unsqueeze(-1), player_id=pid)
    score = score_candidates(status, prod=prod, alive_by_step=alive_by_step,
                             player_count=pc, launches=launches, player_id=pid)
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    pids = obs_tensors["planets"][..., 0].long()
    cand_tgt_pid = pids[cand_tgt.clamp(0, P - 1)]
    return cand_tgt_pid.tolist(), score, cand_valid


def _inject_opponent_expansion(mov, obs, obs_tensors, pid, pc):
    """Each opponent: its strongest 2 planets each launch safe_drain at their
    nearest neutral (mirror of our own expansion logic). Injects their arrivals."""
    P = obs.P; dev = obs.device; dt = obs.ships.dtype
    pids = obs_tensors["planets"][..., 0].long()
    for opp in range(pc):
        if opp == pid:
            continue
        opp_mask = obs.alive & (obs.owner_abs == float(opp)) & (obs.ships >= 5.0)
        if not bool(opp_mask.any()):
            continue
        opp_idx, opp_exists = _candidate_indices(obs.ships, opp_mask, 2)
        neutral_mask = obs.alive & (obs.owner_abs < 0)
        srcs, angs, shps, vals = [], [], [], []
        for si in range(int(opp_idx.shape[0])):
            if not bool(opp_exists[si]):
                continue
            s = int(opp_idx[si])
            sx, sy = float(obs.x[s]), float(obs.y[s])
            # nearest neutral
            best = None; bestd = 1e9
            for t in range(P):
                if not bool(neutral_mask[t]):
                    continue
                d = ((float(obs.x[t]) - sx) ** 2 + (float(obs.y[t]) - sy) ** 2) ** 0.5
                if d < bestd:
                    bestd = d; best = t
            if best is None:
                continue
            ships = float(obs.ships[s])
            aim = intercept_angle(mov, torch.tensor([s]), torch.tensor([best]),
                                  torch.tensor([ships], dtype=dt))
            if not bool(aim["viable"][0]):
                continue
            srcs.append(s); angs.append(float(aim["angle"][0])); shps.append(ships); vals.append(True)
        if not srcs:
            continue
        ent = LaunchEntries(
            source_slots=torch.tensor(srcs, dtype=torch.long, device=dev),
            target_slots=torch.zeros(len(srcs), dtype=torch.long, device=dev),
            ships=torch.tensor(shps, dtype=dt, device=dev),
            angle=torch.tensor(angs, dtype=dt, device=dev),
            eta=torch.ones(len(srcs), dtype=dt, device=dev),
            valid=torch.tensor(vals, dtype=torch.bool, device=dev))
        pl = infer_planned_launches_from_entries(obs_tensors=obs_tensors, movement=mov, entries=ent, player_id=opp)
        apply_private_planned_launches(movement=mov, launches=pl, owner_id=opp, obs_tensors=obs_tensors)


def rank_compare(tids, sa, va, sb, vb):
    import math
    fa = [(t, float(s)) for t, s, v in zip(tids, sa.tolist(), va.tolist()) if v and math.isfinite(s)]
    fb = [(t, float(s)) for t, s, v in zip(tids, sb.tolist(), vb.tolist()) if v and math.isfinite(s)]
    if not fa:
        return None
    # rank by score desc
    ra = [t for t, _ in sorted(fa, key=lambda x: -x[1])]
    rb = [t for t, _ in sorted(fb, key=lambda x: -x[1])]
    top3a, top3b = ra[:3], rb[:3]
    argmax_changed = (ra[0] != rb[0]) if rb else True
    # roi crossing: candidates with score>1.55
    setA = {t for t, s in fa if s > 1.55}
    setB = {t for t, s in fb if s > 1.55}
    crossed = setA.symmetric_difference(setB)
    return dict(nA=len(fa), nB=len(fb), argmax_changed=argmax_changed,
                top3_overlap=len(set(top3a) & set(top3b)),
                roi_set_changed=len(crossed), best_A=ra[0], best_B=(rb[0] if rb else None))


def main():
    OPP = ["opponents/structured.py", "opponents/proto1000.py", "opponents/orbitbotnext.py"]
    frames = []
    for seed in [2, 3, 5]:
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        env.run(["single-size/main.py"] + OPP)
        st = env.steps
        for t in [25, 40, 60, 80]:
            if t < len(st):
                ob = st[t][0]["observation"]
                if ob.get("planets"):
                    frames.append((seed, t, dict(ob)))
    print(f"frames: {len(frames)}")
    changed = 0
    for seed, t, ob in frames:
        from orbit_lite.adapter import single_obs_to_tensor
        ot = single_obs_to_tensor(ob, player_id=0)
        tids, sA, vA = gen_and_score(ot, react=False)
        _, sB, vB = gen_and_score(ot, react=True)
        if len(tids) == 0:
            continue
        r = rank_compare(tids, sA, vA, sB, vB)
        if r is None:
            continue
        flag = r["argmax_changed"] or r["top3_overlap"] < 3 or r["roi_set_changed"] > 0
        changed += int(flag)
        print(f"seed{seed} t{t}: cands={r['nA']} argmax_chg={r['argmax_changed']} "
              f"top3_overlap={r['top3_overlap']}/3 roi_set_chg={r['roi_set_changed']} "
              f"best {r['best_A']}->{r['best_B']}  {'<== RANK CHANGED' if flag else ''}")
    print(f"\nframes with rank/roi change: {changed}/{len(frames)}")
    print("==> many changed: reactive opp model IS viable (changes relative ordering).")
    print("==> ~zero changed: v115 -- abort.")


if __name__ == "__main__":
    main()
