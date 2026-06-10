"""Deployable value-in-search agent (direction-3).

Base = exp48 producer (candidate gen + multi-size + endgame). Each turn it asks
the producer for a full-turn action under K aggression configs, projects each
action's resulting state to the horizon via orbit_lite (do-nothing projection),
extracts the SAME features used in training, scores with the learned value V
(P(win)), and emits the highest-V action. V is trained on real game outcomes, so
it values long-horizon position (snowball / hoard / over-extension) that the
myopic flow-diff cannot see.

Deploy bundle: this file AS main.py + main_exp48.py + orbit_lite/ + the value
weights json (path via env VALUE_WEIGHTS or alongside as value.json).

If weights are missing it degrades to plain exp48 (the strongest hand agent),
so it is never worse than baseline by construction.
"""
from __future__ import annotations
import os, sys, json, dataclasses

_HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
for p in (_HERE, os.path.dirname(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch

# exp48 producer building blocks
import importlib.util as _ilu
def _load_mod(name, path):
    spec = _ilu.spec_from_file_location(name, path); m = _ilu.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

_EXP48_PATH = None
for cand in ("main_exp48.py", os.path.join("single-size", "main_exp48.py"),
             os.path.join(os.path.dirname(_HERE), "single-size", "main_exp48.py")):
    if os.path.exists(cand):
        _EXP48_PATH = cand; break
if _EXP48_PATH is None and os.path.exists(os.path.join(_HERE, "main_exp48.py")):
    _EXP48_PATH = os.path.join(_HERE, "main_exp48.py")
P = _load_mod("exp48_base", _EXP48_PATH)

from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.movement_step import (
    ensure_planet_movement, disambiguate_duplicate_launches,
    infer_planned_launches_from_entries, apply_private_planned_launches,
)
from orbit_lite.planner_core import entries_to_sparse_payload, empty_action_row
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

sys.path.insert(0, os.path.join(_HERE))
from features import extract_features, N_FEATURES   # rl/features.py
import value_net as VN

# ---- load value weights (optional) ----------------------------------------
_W = None
for wp in (os.environ.get("VALUE_WEIGHTS"), os.path.join(_HERE, "value.json"),
           "value.json", os.path.join(_HERE, "data", "value_latest.json")):
    if wp and os.path.exists(wp):
        try:
            _W = VN.load_weights(wp); break
        except Exception:
            pass

# aggression config variants to choose among (roi, max_waves multipliers)
_VARIANTS = [
    dict(roi_threshold=1.0, label="aggr"),
    dict(roi_threshold=1.5, label="base"),
    dict(roi_threshold=2.5, label="hold"),   # conservative / hoard
]


def _project_features(movement, obs_tensors, entries, pid, nP, H):
    """Apply entries transiently, project garrison_status to H, build pseudo-obs
    planets, return training features. Restores movement after."""
    snap = {k: (None if getattr(movement, k) is None else getattr(movement, k).clone())
            for k in ("fleet_buckets", "garrison_owner_cache", "garrison_ships_cache",
                      "garrison_pre_combat_owner_cache", "garrison_pre_combat_ships_cache",
                      "garrison_dirty_from")}
    try:
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=pid)
        apply_private_planned_launches(movement=movement, launches=launches,
                                       owner_id=pid, obs_tensors=obs_tensors)
        st = movement.garrison_status(max_horizon=H)
        owner = st.owner[:, -1].tolist()      # owner at horizon
        ships = st.ships[:, -1].tolist()
    finally:
        for k, v in snap.items():
            setattr(movement, k, v)
    planet_ids = obs_tensors["planets"][..., 0].long().tolist()
    prod = movement.planet_prod.tolist()
    x = movement.x[0].tolist(); y = movement.y[0].tolist()
    planets = []
    for i, pidd in enumerate(planet_ids):
        if pidd < 0:
            continue
        planets.append([pidd, int(owner[i]), x[i], y[i], 0.0, float(ships[i]), float(prod[i])])
    pseudo = {"planets": planets, "step": int(obs_tensors["step"].reshape(-1)[0].item()) + H}
    return extract_features(pseudo, pid, nP)


def _entries_for_config(obs_tensors, base_cfg, variant, player_count, memory):
    cfg = dataclasses.replace(base_cfg, **{k: v for k, v in variant.items() if k != "label"})
    obs = parse_obs(obs_tensors)
    if obs.P == 0:
        return None, None, None
    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=P._movement_config(cfg, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None))
    memory.movement = movement
    cache = build_distance_cache(movement, max_k=int(cfg.horizon))
    H = int(cfg.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]
    entries = P.plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=cfg, player_count=int(player_count))
    entries = disambiguate_duplicate_launches(entries)
    return entries, movement, (obs, H)


class _Mem:
    def __init__(self): self.movement = None; self.pc = None


_RT = _Mem()


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    pid = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=pid)
    with torch.no_grad():
        if bool((obs_tensors["step"] == 0).all()):
            _RT.pc = None
        if _RT.pc is None:
            from orbit_lite.planner_core import largest_initial_player_count
            _RT.pc = largest_initial_player_count(obs_tensors)
        nP = int(_RT.pc)
        base = P._config_for(nP)
        step = int(obs_tensors["step"].reshape(-1)[0].item())
        base = P._apply_phase_config(base, step)

        # No value weights -> behave exactly like exp48 (never worse than baseline).
        if _W is None:
            return P.agent(obs)

        best_payload = None; best_v = -1.0
        for variant in _VARIANTS:
            entries, movement, ctx = _entries_for_config(obs_tensors, base, variant, nP, _RT)
            if entries is None:
                continue
            obs_p, H = ctx
            feats = _project_features(movement, obs_tensors, entries, pid, nP, H)
            v = VN.forward_prob(_W, feats)
            if v > best_v:
                best_v = v
                planet_ids = obs_tensors["planets"][..., 0].long()
                # re-apply chosen entries to movement for next-turn tracking
                best_payload = entries_to_sparse_payload(entries, planet_ids=planet_ids)
                best_entries = entries
        if best_payload is None:
            return P.agent(obs)
        # commit chosen entries into the rolling cache
        launches = infer_planned_launches_from_entries(
            obs_tensors=obs_tensors, movement=_RT.movement, entries=best_entries, player_id=pid)
        apply_private_planned_launches(movement=_RT.movement, launches=launches,
                                       owner_id=pid, obs_tensors=obs_tensors)
        return sparse_action_row_to_moves(best_payload, obs, player_id=pid)
