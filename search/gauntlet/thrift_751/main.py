from __future__ import annotations

import dataclasses
import os
import sys
from dataclasses import dataclass

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import torch
from torch import Tensor

from orbit_lite.geometry import fleet_speed
from orbit_lite.intercept_aim import intercept_angle
from orbit_lite.movement import MovementConfig, PlanetMovement
from orbit_lite.movement_step import (
    apply_private_planned_launches,
    concat_launch_entries,
    disambiguate_duplicate_launches,
    ensure_planet_movement,
    infer_planned_launches_from_entries,
)
from orbit_lite.obs import parse_obs
from orbit_lite.distance_cache import build_distance_cache
from orbit_lite.planner_core import (
    _candidate_indices,
    _empty_entries,
    _greedy_select,
    _plan_regroup,
    build_target_shortlist,
    capture_floor,
    empty_action_row,
    entries_to_sparse_payload,
    largest_initial_player_count,
    make_launch_set,
    reachable_mask,
    reinforcement_timing_factor,
    safe_drain,
    score_candidates,
)
from orbit_lite.adapter import single_obs_to_tensor, sparse_action_row_to_moves

TOTAL_STEPS = 500


@dataclass(frozen=True)
class ProducerLiteConfig:
    """Behaviour knobs."""

    horizon: int = 18
    max_sources_per_lane: int = 12
    max_offensive_targets: int = 12
    max_defensive_targets: int = 4
    max_waves_per_turn: int = 7
    roi_threshold: float = 1.5
    min_ships_to_launch: float = 4.0
    reinforce_size_beta: float = 2.2
    reinforce_eta_free: float = 3.0
    reinforce_eta_scale: float = 12.0
    enable_regroup: bool = True
    max_regroup_time: float = 6.0
    regroup_pressure_delta_min: float = 0.35
    max_regroup_sources_per_lane: int = 6
    max_regroup_targets_per_source: int = 5
    regroup_pressure_norm: str = "none"
    regroup_time_penalty_weight: float = 1e-3
    terminal_phase_turns: int = 40
    terminal_roi_threshold: float = 1.0
    terminal_max_waves_per_turn: int = 8
    terminal_enable_regroup: bool = False
    # --- exp41: evaluate several commit fractions per (src, tgt) ------------
    size_multipliers: tuple[float, ...] = (0.5, 0.75, 1.0)
    # --- hoard mode (direction: top players accumulate huge garrisons) ------
    # Once we hold >= hoard_min_planets, raise the firing bar so production piles
    # into garrison (win the long ship-weighted game) instead of being spent on
    # marginal attacks. 0 = disabled. Activates only AFTER the base is built, so
    # it does not freeze early expansion (unlike v111 reserve-from-start).
    hoard_min_planets: int = 0
    hoard_roi_mult: float = 1.0      # multiply roi_threshold when hoarding
    hoard_max_waves: int = 0         # cap waves when hoarding (0 = unchanged)
    # --- continuous dynamic control (ported from light-ver 1200) -------------
    # CONTINUOUS strength-ratio -> roi/waves (no hard thresholds, never freezes).
    # behind -> lower roi (attack harder to catch up) via quadratic deficit curve.
    enable_dynamic_roi: bool = False
    # late-game candidate suppression: kill attacks that arrive too late to pay
    # off + depreciate late neutrals (reduce end-game ship waste).
    enable_late_suppress: bool = False
    # --- anti-focus-fire defense (vs >1400: we expand then get wiped) --------
    # Reserve defensive garrison on a source PROPORTIONAL to reachable enemy mass
    # near it (not blanket like v111): rear planets (low enemy_mass) drain fully
    # and expand; threatened frontier planets keep ships to survive a coordinated
    # counter-strike. 0 = disabled.
    defense_reserve_beta: float = 0.0
    # --- ported from i-the-orbit (proactive defense / economy rush / centrality) ---
    # proactive defense: predict owned planets that will FALL (garrison_status
    # ships<0) and send a rescue wave from nearest in-time friend (the active
    # counter to the >1400 "expand then get wiped" loss).
    enable_defense: bool = False
    defense_threat_horizon: float = 14.0
    defense_min_intercept_margin: float = 1.05
    defense_max_waves: int = 3
    # economy rush: early game, boost score of candidates taking top-prod neutrals.
    enable_prod_rush: bool = False
    prod_rush_steps: int = 120
    prod_rush_top_k: int = 3
    prod_rush_roi_discount: float = 0.80
    # orbital centrality: weight source selection toward central hub planets (0=off).
    geometry_weight: float = 0.0
    # --- conditional consolidate vs a strong economy/hoard opponent ----------
    # Detect (my_strength/leader_strength < ratio) i.e. a notably stronger econ
    # rival exists -> switch to "take fewer but hold stably": raise roi (selective
    # expansion) + reserve garrison on threatened planets (PASSIVE hold, not the
    # io ship-spending rescue). Off vs weak opponents -> keep rmargin aggression.
    consolidate_on_strong: bool = False
    consolidate_ratio: float = 0.85
    consolidate_roi_mult: float = 1.5
    consolidate_def_beta: float = 0.6


def _movement_config(config: ProducerLiteConfig, *, player_count: int) -> MovementConfig:
    return MovementConfig(
        movement_horizon=int(config.horizon),
        drift_epsilon=1e-3,
        track_fleets=True,
        player_count=int(player_count),
        max_tracked_fleets=128,
    )


def _apply_phase_config(config: ProducerLiteConfig, step: int) -> ProducerLiteConfig:
    if int(step) >= TOTAL_STEPS - int(config.terminal_phase_turns):
        return dataclasses.replace(
            config,
            roi_threshold=float(config.terminal_roi_threshold),
            max_waves_per_turn=int(config.terminal_max_waves_per_turn),
            enable_regroup=bool(config.terminal_enable_regroup),
        )
    return config


def cheap_enemy_pressure(obs, cache, *, horizon: float, player_id: int) -> Tensor:
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    if P == 0:
        return torch.zeros(P, dtype=dtype, device=device)
    d0 = cache.cross_dist[0].to(dtype)
    ships = obs.ships.to(dtype)
    speeds = fleet_speed(ships.clamp(min=1e-6))
    reach_dist = (speeds.view(P, 1) * float(horizon)).clamp(min=1e-6)
    enemy = obs.alive & (obs.owner_abs >= 0) & (obs.owner_abs != int(player_id))
    eye = torch.eye(P, device=device, dtype=torch.bool)
    valid = enemy.view(P, 1) & obs.alive.view(1, P) & ~eye
    decay = (1.0 - d0 / reach_dist).clamp(min=0.0)
    contrib = torch.where(valid, ships.view(P, 1) * decay, torch.zeros_like(decay))
    return contrib.sum(dim=0)


def _tier_candidates(
    *,
    movement: PlanetMovement,
    source_idx: Tensor,
    source_exists: Tensor,
    target_idx: Tensor,
    target_exists: Tensor,
    target_is_mine: Tensor,
    drain: Tensor,
    floor: Tensor,
    eta_cap: Tensor,
    size_mult: float,
    S: int,
    T: int,
    pid: int,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    player_count: int,
    device,
    dtype,
):
    """Build scored candidates for one fleet-size fraction."""
    sizes = (drain.view(S, 1) * float(size_mult)).floor().clamp(min=1.0).expand(S, T)
    K = int(floor.shape[-1])

    active = reachable_mask(
        movement, source_idx=source_idx, target_idx=target_idx,
        fleet_sizes=sizes.unsqueeze(-1), eta_cap=eta_cap,
    ).squeeze(-1)
    aim = intercept_angle(
        movement,
        source_idx.unsqueeze(1),
        target_idx.unsqueeze(0),
        sizes,
        active=active,
    )
    angle = aim["angle"]
    eta = aim["eta"]
    viable = aim["viable"] & (eta <= eta_cap.view(1, T))

    if K > 0:
        k_arr = (eta.clamp(min=1.0, max=float(K)).ceil().long() - 1).clamp(0, K - 1)
        floor_at_arr = floor.unsqueeze(0).expand(S, T, K).gather(-1, k_arr.unsqueeze(-1)).squeeze(-1)
    else:
        floor_at_arr = torch.ones(S, T, dtype=dtype, device=device)
    clears_floor = sizes >= floor_at_arr

    src_neq_tgt = source_idx.view(S, 1) != target_idx.view(1, T)
    valid = (
        viable & clears_floor & (sizes >= 1.0) & src_neq_tgt
        & source_exists.view(S, 1) & target_exists.view(1, T)
    )

    L = 1
    C = S * T
    cand_src = source_idx.view(S, 1).expand(S, T).reshape(C, L)
    cand_tgt_slot = target_idx.view(1, T).expand(S, T).reshape(C)
    cand_tgt_short = torch.arange(T, device=device).view(1, T).expand(S, T).reshape(C)
    cand_send = torch.where(valid, sizes, torch.zeros_like(sizes)).reshape(C, L)
    cand_angle = angle.reshape(C, L)
    cand_eta = torch.where(valid, eta, torch.ones_like(eta)).reshape(C, L)
    cand_active = valid.reshape(C, L)
    cand_valid = valid.reshape(C)
    cand_is_def = target_is_mine[cand_tgt_short]

    launches = make_launch_set(
        source_slots=cand_src,
        target_slots=cand_tgt_slot.unsqueeze(-1).expand(C, L),
        ships=cand_send,
        eta=cand_eta,
        valid=cand_active & cand_valid.unsqueeze(-1),
        player_id=pid,
    )
    score = score_candidates(
        garrison_status, prod=prod, alive_by_step=alive_by_step,
        player_count=int(player_count), launches=launches, player_id=pid,
    )
    score = torch.where(cand_valid, score, torch.full_like(score, float("-inf")))
    return cand_src, cand_send, cand_angle, cand_eta, cand_active, cand_tgt_slot, cand_tgt_short, cand_is_def, score


def plan_lite_waves(
    *,
    movement: PlanetMovement,
    obs,
    obs_tensors: dict,
    cache,
    garrison_status,
    prod: Tensor,
    alive_by_step: Tensor,
    config: ProducerLiteConfig,
    player_count: int,
):
    P = obs.P
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    H_axis = int(garrison_status.ships.shape[-1])
    H = max(H_axis - 1, 0)
    K_eta = max(1, min(int(config.horizon), H))
    W = max(1, int(config.max_waves_per_turn))

    source_mask = obs.owned & obs.alive & (obs.ships >= float(config.min_ships_to_launch))
    if not bool(source_mask.any()):
        return _empty_entries(device, dtype)

    S_cap = max(1, min(int(config.max_sources_per_lane), P))
    if float(getattr(config, "geometry_weight", 0.0)) > 0.0:
        gw = float(config.geometry_weight)
        centrality = _orbital_centrality(obs, cache)
        ships_f = obs.ships.to(dtype)
        prod_f = prod.to(dtype)
        src_score = ((1.0 - gw) * (ships_f + 0.5 * prod_f * (ships_f / (ships_f + 1.0)))
                     + gw * centrality * ships_f)
        src_score = torch.where(source_mask, src_score,
                                torch.tensor(float("-inf"), device=device, dtype=dtype))
        source_idx = torch.topk(src_score, min(S_cap, int(src_score.numel())), dim=0).indices
        source_exists = source_mask[source_idx]
    else:
        source_idx, source_exists = _candidate_indices(obs.ships, source_mask, S_cap)
    target_idx, target_exists = build_target_shortlist(
        obs, obs_tensors, garrison_status, cache,
        config=config, K_eta=K_eta, H=H, prod=prod, source_mask=source_mask,
    )
    if not bool(target_exists.any()):
        return _empty_entries(device, dtype)
    S = int(source_idx.shape[0])
    T = int(target_idx.shape[0])
    target_is_mine = obs.owned[target_idx.clamp(0, P - 1)]

    source_ships = obs.ships[source_idx.clamp(0, P - 1)].to(dtype)
    H_eff = torch.full((), float(H), dtype=dtype, device=device)
    drain = safe_drain(
        garrison_status, source_idx=source_idx, source_ships=source_ships,
        H_eff=H_eff, player_id=pid,
    )

    eta_cap = torch.full((T,), float(K_eta), dtype=dtype, device=device)
    beta = float(config.reinforce_size_beta)
    def_beta = float(getattr(config, "defense_reserve_beta", 0.0))
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        if beta > 0.0 or def_beta > 0.0 or bool(config.enable_regroup) else None
    )
    # anti-focus-fire: hold ships on threatened frontier planets (reserve ∝ reachable
    # enemy mass near the source). Rear planets keep draining/expanding (reserve≈0).
    if def_beta > 0.0 and enemy_mass is not None:
        reserve = (def_beta * enemy_mass[source_idx.clamp(0, P - 1)]).clamp(min=0.0)
        drain = (drain - reserve).clamp(min=0.0).floor()
    reinforcement = None
    if beta > 0.0:
        enemy_mass_t = enemy_mass[target_idx.clamp(0, P - 1)]
        k_arange = torch.arange(1, K_eta + 1, device=device, dtype=dtype)
        rho = reinforcement_timing_factor(
            k_arange,
            eta_free=float(config.reinforce_eta_free),
            eta_scale=float(config.reinforce_eta_scale),
        )
        reinforcement = beta * rho.view(1, K_eta) * enemy_mass_t.view(T, 1)
    floor = capture_floor(
        garrison_status, target_idx=target_idx, k_max=K_eta,
        capture_overhead=1.0, player_id=pid, reinforcement=reinforcement,
    )

    tier_parts = [
        _tier_candidates(
            movement=movement,
            source_idx=source_idx,
            source_exists=source_exists,
            target_idx=target_idx,
            target_exists=target_exists,
            target_is_mine=target_is_mine,
            drain=drain,
            floor=floor,
            eta_cap=eta_cap,
            size_mult=float(mult),
            S=S,
            T=T,
            pid=pid,
            garrison_status=garrison_status,
            prod=prod,
            alive_by_step=alive_by_step,
            player_count=player_count,
            device=device,
            dtype=dtype,
        )
        for mult in config.size_multipliers
    ]

    cand_src = torch.cat([p[0] for p in tier_parts], dim=0)
    cand_send = torch.cat([p[1] for p in tier_parts], dim=0)
    cand_angle = torch.cat([p[2] for p in tier_parts], dim=0)
    cand_eta = torch.cat([p[3] for p in tier_parts], dim=0)
    cand_active = torch.cat([p[4] for p in tier_parts], dim=0)
    cand_tgt_slot = torch.cat([p[5] for p in tier_parts], dim=0)
    cand_tgt_short = torch.cat([p[6] for p in tier_parts], dim=0)
    cand_is_def = torch.cat([p[7] for p in tier_parts], dim=0)
    score = torch.cat([p[8] for p in tier_parts], dim=0)

    if bool(getattr(config, "enable_late_suppress", False)):
        _step = int(obs_tensors["step"].reshape(-1)[0].item())
        score = _suppress_late_candidates(
            score=score, obs=obs, target_idx=target_idx, cand_tgt_short=cand_tgt_short,
            cand_is_def=cand_is_def, cand_eta=cand_eta, step=_step, player_id=pid,
        )
    if bool(getattr(config, "enable_prod_rush", False)):
        _step = int(obs_tensors["step"].reshape(-1)[0].item())
        score = _apply_prod_snowball_boost(
            score=score, obs=obs, target_idx=target_idx, cand_tgt_short=cand_tgt_short,
            prod=prod, step=_step, config=config,
        )

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if not bool(config.enable_regroup):
        return wave_entries
    if enemy_mass is None:
        enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
    regroup_entries = _plan_regroup(
        movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
        leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
        config=config, H=H,
    )
    return concat_launch_entries([wave_entries, regroup_entries])


def _apply_hoard_config(config: ProducerLiteConfig, owned_planets: int) -> ProducerLiteConfig:
    """When the base is built (owned >= hoard_min_planets), raise the firing bar
    so production accumulates into garrison instead of marginal attacks."""
    if int(config.hoard_min_planets) <= 0 or owned_planets < int(config.hoard_min_planets):
        return config
    repl = {}
    if float(config.hoard_roi_mult) != 1.0:
        repl["roi_threshold"] = float(config.roi_threshold) * float(config.hoard_roi_mult)
    if int(config.hoard_max_waves) > 0:
        repl["max_waves_per_turn"] = int(config.hoard_max_waves)
    return dataclasses.replace(config, **repl) if repl else config


def _owner_strength(obs, prod: Tensor, player_count: int) -> Tensor:
    """Per-owner production + 2.5% ships as strength proxy. [player_count]"""
    device = obs.device
    dtype = obs.ships.dtype
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    owner = obs.owner_abs.to(torch.long)
    prod_v = prod.to(dtype)
    ships = obs.ships.to(dtype)
    for oid in range(int(player_count)):
        mask = obs.alive & (owner == oid)
        if bool(mask.any()):
            strength[oid] = prod_v[mask].sum() + 0.025 * ships[mask].sum()
    return strength


def _adjust_config(config, obs, prod, step: int, player_count: int):
    """CONTINUOUS strength-ratio -> roi/waves (no hard thresholds). Behind ->
    lower roi (attack harder) via quadratic deficit curve + late-game urgency."""
    pid = int(obs.player_id)
    strength = _owner_strength(obs, prod, int(player_count))
    if pid < 0 or pid >= int(player_count) or strength.numel() == 0:
        return config
    my = float(strength[pid].item()); leader = float(strength.max().item())
    ratio = my / max(leader, 1e-6)
    if ratio < 1.0:
        deficit = 1.0 - ratio
        roi_drop = 0.25 * deficit * deficit
        new_roi = max(1.10, float(config.roi_threshold) - roi_drop)
        remaining = TOTAL_STEPS - int(step)
        if remaining < 150 and ratio < 0.90:
            urgency = (150 - remaining) / 150.0
            new_roi = max(1.10, new_roi - 0.10 * urgency * deficit)
        config = dataclasses.replace(config, roi_threshold=new_roi)
    waves = int(config.max_waves_per_turn)
    if ratio < 0.70:
        waves = min(8, waves + 1)
    if (TOTAL_STEPS - int(step)) < 100 and ratio < 0.95:
        waves = min(8, waves + 1)
    if waves != int(config.max_waves_per_turn):
        config = dataclasses.replace(config, max_waves_per_turn=waves)
    return config


def _suppress_late_candidates(*, score, obs, target_idx, cand_tgt_short, cand_is_def,
                              cand_eta, step: int, player_id: int):
    """Last 120 turns: kill attacks arriving too late to pay off; depreciate late neutrals."""
    remaining = TOTAL_STEPS - int(step)
    if remaining > 120 or int(obs.P) <= 0 or score.numel() == 0:
        return score
    device = score.device; dtype = score.dtype; pid = int(player_id); P = int(obs.P)
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs.to(device=device)[tgt_abs].long()
    eta = cand_eta.reshape(score.shape).to(device=device, dtype=dtype)
    is_neutral = tgt_owner < 0
    is_enemy = (tgt_owner >= 0) & (tgt_owner != pid) & (~cand_is_def)
    neutral_margin = max(1.0, float(remaining) - 8.0)
    enemy_margin = max(1.0, float(remaining) - 4.0)
    too_late = (is_neutral & (eta > neutral_margin)) | (is_enemy & (eta > enemy_margin))
    neutral_factor = ((float(remaining) - eta) / max(1.0, 80.0)).clamp(min=0.20, max=1.0)
    score = torch.where(is_neutral, score * neutral_factor, score)
    return torch.where(too_late, torch.full_like(score, float("-inf")), score)


def _orbital_centrality(obs, cache) -> Tensor:
    """Return a [P] tensor: higher = more central in the map."""
    P = int(obs.P)
    device = obs.device
    if P <= 1:
        return torch.ones(P, device=device)
    d0 = cache.cross_dist[0].clone().float()              # [P, P]
    alive = obs.alive.to(device=device)
    # Zero out distances to dead planets so they don't pollute the mean
    d0 = torch.where(alive.view(1, P) & alive.view(P, 1), d0, torch.zeros_like(d0))
    n_alive = alive.float().sum().clamp(min=1.0)
    mean_dist = d0.sum(dim=1) / n_alive                   # [P]  avg dist to all alive
    # Invert: closer on average → higher centrality
    centrality = 1.0 / (mean_dist + 1.0)
    return centrality.to(obs.ships.dtype)


# ---------------------------------------------------------------------------
# NEW: Proactive defense – detect enemy fleets heading for our planets and
# reinforce those planets BEFORE they fall.
#
# Strategy:
#   For each owned planet that has inbound enemy ships arriving in < horizon:
#     If current_ships + inbound_friendly < enemy_ships_arriving:
#       Launch a reinforcement wave from the nearest friendly planet that can
#       still arrive in time.
# ---------------------------------------------------------------------------


def _build_defense_entries(
    *,
    movement: PlanetMovement,
    obs,
    cache,
    config: ProducerLiteConfig,
    player_count: int,
):
    """Build emergency reinforcement entries for threatened planets."""
    P = int(obs.P)
    device = obs.device
    dtype = obs.ships.dtype
    pid = int(obs.player_id)

    if P == 0:
        return _empty_entries(device, dtype)

    owned = obs.owned & obs.alive                         # [P] bool – my alive planets
    if not bool(owned.any()):
        return _empty_entries(device, dtype)

    # ------------------------------------------------------------------
    # Step 1: estimate net balance on each owned planet over the defense horizon
    # We use garrison_status which already tracks inbound fleets by step.
    H = min(int(config.defense_threat_horizon), int(movement.garrison_status(max_horizon=int(config.defense_threat_horizon)).ships.shape[-1]) - 1)
    if H <= 0:
        return _empty_entries(device, dtype)

    status = movement.garrison_status(max_horizon=H)
    # ships[p, h] = expected ship count at planet p at step h
    # Look at step H (the end of our threat window)
    ships_at_H = status.ships[:, -1]   # [P]

    # Planets that are ours now but are projected to fall (ships_at_H < 0)
    threatened = owned & (ships_at_H < 0)
    if not bool(threatened.any()):
        return _empty_entries(device, dtype)

    # ------------------------------------------------------------------
    # Step 2: for each threatened planet, find the best rescue source
    tgt_indices = threatened.nonzero(as_tuple=False).squeeze(1)   # [T_def]
    src_indices = owned.nonzero(as_tuple=False).squeeze(1)         # [S]

    if src_indices.numel() == 0 or tgt_indices.numel() == 0:
        return _empty_entries(device, dtype)

    d0 = cache.cross_dist[0].to(dtype)    # [P, P]
    src_ships = obs.ships[src_indices].to(dtype)  # [S]

    all_entries = []
    waves_launched = 0

    for t_i in range(int(tgt_indices.shape[0])):
        if waves_launched >= int(config.defense_max_waves):
            break
        tgt = int(tgt_indices[t_i].item())
        deficit = float(-ships_at_H[tgt].item())          # How many ships we need
        need = deficit * float(config.defense_min_intercept_margin)

        # Find sources that have surplus ships and can reach in time
        # Speed approximation: fleet_speed depends on ship count (orbit_lite convention)
        dists = d0[src_indices, tgt]                       # [S]
        speeds = fleet_speed(src_ships.clamp(min=1.0))    # [S]
        etas = (dists / speeds.clamp(min=1e-6)).ceil()    # [S] steps to arrive

        can_arrive = etas <= float(H)
        has_surplus = src_ships > (need + float(config.min_ships_to_launch))
        src_neq_tgt = src_indices != tgt
        valid_src = can_arrive & has_surplus & src_neq_tgt

        if not bool(valid_src.any()):
            continue

        # Prefer closest valid source
        best_src_local = int(torch.where(valid_src, dists, torch.full_like(dists, 1e9)).argmin().item())
        best_src = int(src_indices[best_src_local].item())
        send_ships = min(float(src_ships[best_src_local].item()) * 0.6,  # send up to 60% garrison
                         need + float(config.min_ships_to_launch))
        send_ships = max(send_ships, float(config.min_ships_to_launch))

        # Build a single-lane entry using make_launch_set
        src_t = torch.tensor([[best_src]], dtype=torch.long, device=device)
        tgt_t = torch.tensor([tgt],       dtype=torch.long, device=device)
        send_t = torch.tensor([[send_ships]], dtype=dtype, device=device)
        eta_t = torch.tensor([[float(etas[best_src_local].item())]], dtype=dtype, device=device)
        valid_t = torch.tensor([[True]], dtype=torch.bool, device=device)

        entry = make_launch_set(
            source_slots=src_t,
            target_slots=tgt_t.unsqueeze(-1).expand(1, 1),
            ships=send_t,
            eta=eta_t,
            valid=valid_t,
            player_id=pid,
        )
        all_entries.append(entry)
        waves_launched += 1

    if not all_entries:
        return _empty_entries(device, dtype)
    return concat_launch_entries(all_entries)


# ---------------------------------------------------------------------------
# Dynamic adjustment – multi-knob (enhanced)
# ---------------------------------------------------------------------------


def _apply_prod_snowball_boost(
    *,
    score: Tensor,
    obs,
    target_idx: Tensor,
    cand_tgt_short: Tensor,
    prod: Tensor,
    step: int,
    config: ProducerLiteConfig,
) -> Tensor:
    if int(step) > int(config.prod_rush_steps):
        return score  # Only active in early game

    P = int(obs.P)
    device = score.device
    dtype = score.dtype

    neutral_mask = obs.owner_abs < 0                    # [P] neutral planets
    if not bool(neutral_mask.any()):
        return score

    prod_neutral = torch.where(neutral_mask & obs.alive, prod.to(dtype), torch.zeros(P, dtype=dtype, device=device))
    if int(prod_neutral.numel()) == 0:
        return score

    # Find the top-K production values among neutrals
    top_k = min(int(config.prod_rush_top_k), int(prod_neutral.numel()))
    top_vals = torch.topk(prod_neutral, top_k).values
    if top_vals.numel() == 0:
        return score
    threshold = float(top_vals[-1].item())             # Minimum qualifying prod value

    # Identify which candidates target a top-prod neutral
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_prod = prod.to(dtype)[tgt_abs]
    tgt_neutral = (obs.owner_abs[tgt_abs] < 0)
    is_top_prod_neutral = tgt_neutral & (tgt_prod >= threshold - 1e-6)

    # Boost: lower effective ROI requirement = divide score threshold; here we
    # multiply the raw score so these candidates survive greedy_select easier.
    boost_factor = 1.0 / float(config.prod_rush_roi_discount)  # e.g. 1/0.80 = 1.25
    score = torch.where(is_top_prod_neutral.reshape(score.shape), score * boost_factor, score)
    return score


# ---------------------------------------------------------------------------
# Core planner – with orbital geometry source selection + snowball scoring
# ---------------------------------------------------------------------------


def _maybe_consolidate(config, obs, prod, player_count: int):
    """If a notably stronger economy opponent exists (my/leader < consolidate_ratio),
    switch to take-fewer-hold-stably: raise roi + reserve garrison (passive hold)."""
    pid = int(obs.player_id)
    strength = _owner_strength(obs, prod, int(player_count))
    if pid < 0 or pid >= int(player_count) or strength.numel() == 0:
        return config
    my = float(strength[pid].item()); leader = float(strength.max().item())
    if my / max(leader, 1e-6) >= float(config.consolidate_ratio):
        return config
    return dataclasses.replace(
        config,
        roi_threshold=float(config.roi_threshold) * float(config.consolidate_roi_mult),
        defense_reserve_beta=max(float(config.defense_reserve_beta), float(config.consolidate_def_beta)),
    )


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)
    config = _apply_hoard_config(config, int(obs.owned.sum().item()))

    movement = ensure_planet_movement(
        obs_tensors=obs_tensors,
        expected_cfg=_movement_config(config, player_count=int(player_count)),
        cached_movement=getattr(memory, "movement", None),
    )
    memory.movement = movement
    if bool(getattr(config, "enable_dynamic_roi", False)):
        _step = int(obs_tensors["step"].reshape(-1)[0].item())
        config = _adjust_config(config, obs, movement.planet_prod, _step, int(player_count))
    if bool(getattr(config, "consolidate_on_strong", False)):
        config = _maybe_consolidate(config, obs, movement.planet_prod, int(player_count))
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
    if bool(getattr(config, "enable_defense", False)):
        defense_entries = _build_defense_entries(
            movement=movement, obs=obs, cache=cache, config=config,
            player_count=int(player_count),
        )
        entries = concat_launch_entries([defense_entries, entries])
    entries = disambiguate_duplicate_launches(entries)
    launches = infer_planned_launches_from_entries(
        obs_tensors=obs_tensors, movement=movement, entries=entries, player_id=int(obs.player_id),
    )
    apply_private_planned_launches(
        movement=movement, launches=launches, owner_id=int(obs.player_id),
        obs_tensors=obs_tensors,
    )
    planet_ids = obs_tensors["planets"][..., 0].long()
    return entries_to_sparse_payload(entries, planet_ids=planet_ids)


CONFIG_4P = dataclasses.replace(
    ProducerLiteConfig(),
    horizon=13,
    max_sources_per_lane=6,
    max_defensive_targets=2,
    max_regroup_time=6.0,
    regroup_pressure_delta_min=0.25,
    max_regroup_targets_per_source=8,
)



# ---------------------------------------------------------------------------
# Search params injected by search/generate_candidates.py
# ---------------------------------------------------------------------------

import json as _ow_json


def _ow_load_strategy_params() -> dict:
    path = os.path.join(_HERE, "params.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = _ow_json.load(handle)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _ow_apply_config_overrides(config: ProducerLiteConfig, values: dict | None) -> ProducerLiteConfig:
    if not values:
        return config
    allowed = {field.name for field in dataclasses.fields(ProducerLiteConfig)}
    cleaned = {}
    for key, value in values.items():
        if key not in allowed:
            continue
        if key == "size_multipliers":
            value = tuple(float(x) for x in value)
        cleaned[key] = value
    return dataclasses.replace(config, **cleaned)


_OW_STRATEGY_PARAMS = _ow_load_strategy_params()
CONFIG_2P = _ow_apply_config_overrides(ProducerLiteConfig(), _OW_STRATEGY_PARAMS.get("config_2p"))
CONFIG_4P = _ow_apply_config_overrides(CONFIG_4P, _OW_STRATEGY_PARAMS.get("config_4p"))


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else CONFIG_2P


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
        if mem.cached_player_count is None:
            mem.cached_player_count = largest_initial_player_count(obs_tensors)
        current_player = int(obs_tensors["player"].reshape(-1)[0].item())
        min_count = current_player + 1
        mem.cached_player_count = 4 if max(int(mem.cached_player_count), min_count) > 2 else 2
        base = _config_for(mem.cached_player_count)
        step = int(obs_tensors["step"].reshape(-1)[0].item())
        config = _apply_phase_config(base, step)
        row = run_turn(
            obs_tensors, config=config,
            player_count=int(mem.cached_player_count), memory=mem,
        )
        mem.last_sparse_action_row = row
        return row


_RUNTIME = ProducerLiteRuntime()


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    player_id = int(player)
    obs_tensors = single_obs_to_tensor(obs, player_id=player_id)
    with torch.no_grad():
        sparse_row = _RUNTIME.tensor_action(obs_tensors)
    return sparse_action_row_to_moves(sparse_row, obs, player_id=player_id)
