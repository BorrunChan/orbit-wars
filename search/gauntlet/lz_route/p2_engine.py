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
from orbit_lite.garrison_launch import sparse_launch_flow_delta
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
    LaunchEntries,
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
    # reactive opponent model: inject predicted opponent launches into the
    # garrison projection so the (otherwise do-nothing) scorer SEES incoming
    # counter-attacks -> won't over-extend into the enemy's reach, defends
    # planets the enemy will hit. Ported from single-size/main_react.py; here
    # ungated so it can run in 2P (the 'beat rmargin' lever). off by default.
    react_opponent: bool = False
    react_sources_per_opp: int = 2
    # evacuation: if a planet WILL fall to in-flight enemy fleets (garrison_status
    # projects its ships negative), ship its whole garrison to the nearest SAFE
    # friendly planet reachable BEFORE it falls -> save the army instead of losing
    # it with the planet; the attacker's overkill fleet then hits an empty planet
    # (out of position). Reacts to fleets already launched, no prediction needed.
    enable_evacuation: bool = False
    evac_min_ships: float = 10.0
    evac_horizon: int = 10
    evac_max_waves: int = 4
    # 1+2 联动 (金蝉脱壳 + 围魏救赵): a doomed planet's garrison, instead of just
    # fleeing to a safe friend, prefers to RETREAT INTO a capture of the
    # attacker's now-drained source planet (retreat == counter-attack). Falls
    # back to fleeing if no winnable counterstrike is reachable.
    evac_counterstrike: bool = True
    evac_counter_margin: float = 1.25
    # last-second only: evac a doomed planet ONLY when it falls within this many
    # steps (default 1 = lands next turn) — i.e. too late for the producer's
    # regroup to defend it during the flight window. Avoids premature abandon of
    # planets the producer would still save. (perfect-info: incoming is KNOWN)
    evac_last_second_steps: int = 1
    # opportunist strike (围魏救赵 / 趁火打劫 / 欲擒故纵): boost capturing ENEMY
    # planets that are currently weak (low garrison) — a drained source that just
    # launched at us, an over-extended planet, or one being eaten by a third
    # party all read as low current ships. Lets the greedy take these targets of
    # opportunity even when just under the ROI bar. off by default.
    enable_comet: bool = False
    comet_prod_bonus: float = 4.0
    enable_opportunist: bool = False
    opportunist_max_garrison: float = 15.0
    opportunist_boost: float = 1.5
    # 围魏救赵 (计1): boost capturing enemy planets that just launched a fleet
    # (their source is now drained). 趁火打劫 (计2): boost capturing enemy planets
    # whose garrison is projected to decline (others are eating them). (计3 =
    # opportunist above = low current garrison / over-extended).
    enable_weiwei: bool = False
    weiwei_boost: float = 1.5
    enable_chenghuo: bool = False
    chenghuo_boost: float = 1.5
    # 擒贼擒王 / 釜底抽薪: in 2P (score = 5*prod + ships) boost capturing the
    # enemy's HIGH-PRODUCTION planets — biggest score swing (+my prod, -their
    # prod) and cuts their economy at the root.
    enable_qinwang: bool = False
    qinwang_min_prod: float = 4.0
    qinwang_boost: float = 1.5
    # 4P VULTURE (不战底盘): by default DON'T attack enemy planets at all (only
    # expand neutrals + defend). Only strike an enemy planet when it matches one
    # of the 四计 triggers (围魏救赵 drained source / 趁火打劫 3rd-party-attacked /
    # 欲擒故纵 strong-opp over-extended / 擒贼擒王 leader economy). Let the other
    # producers slaughter each other; vulture the openings. (non-producer 4P)
    enable_vulture: bool = False
    # 4P prod-alignment (framework): 4P score = prod ONLY (ships weight 0), but the
    # flow-diff scorer optimizes net-ship-delta (produced - lost_to_combat). Set
    # combat_loss_weight < 1 to DOWN-weight combat losses so the agent values
    # capturing prod/planets over preserving ships — aligned with the 4P objective.
    # 1.0 = original net-ship-delta (unchanged).
    combat_loss_weight: float = 1.0
    # soft version: instead of -inf (full block), multiply non-四计 enemy-attack
    # scores by this factor. 0.0 = full pacifist block; 0.4-0.7 = "passive base"
    # (base still contests strong enemy planets, but defers marginal enemy attacks
    # to the 四计 precision strikes). Lets the producer base co-exist with tactics.
    vulture_penalty: float = 0.0


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
    config=None,
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
    clw = float(getattr(config, "combat_loss_weight", 1.0))
    if clw != 1.0:
        # prod-aligned: net = produced_delta - clw * lost_combat_delta (down-weight
        # combat loss for 4P where ships don't score). Then me - Σ opp.
        diff = sparse_launch_flow_delta(
            garrison_status, prod=prod, alive_by_step=alive_by_step,
            player_count=int(player_count), launches=launches, player_id=pid,
        )
        net = diff.ships_produced_delta - clw * diff.ships_lost_combat_delta
        me = net[..., pid]
        score = me - (net.sum(dim=-1) - me)
    else:
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

    # --- comet 跳板 (同 p_comet): boost 估值过 roi + 强制进 target 绕 proximity ---
    _comet_idx = None
    if bool(getattr(config, "enable_comet", False)):
        cids = obs_tensors.get("comet_planet_ids")
        if cids is not None and cids.numel() > 0:
            cids = cids.reshape(-1); cids = cids[cids >= 0]
            if cids.numel() > 0:
                pl_ids = obs_tensors["planets"][..., 0].long().reshape(-1)
                cm = (pl_ids.unsqueeze(1) == cids.unsqueeze(0).long()).any(dim=1) & (~obs.owned)
                if bool(cm.any()):
                    _comet_idx = cm.nonzero(as_tuple=False).squeeze(1)
                    prod = prod.clone()
                    prod[cm] = prod[cm] + float(config.comet_prod_bonus)

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
    if _comet_idx is not None and _comet_idx.numel() > 0:
        existing = set(target_idx.tolist())
        extra = [int(c) for c in _comet_idx.tolist() if int(c) not in existing]
        if extra:
            et = torch.tensor(extra, dtype=target_idx.dtype, device=device)
            target_idx = torch.cat([target_idx, et])
            target_exists = torch.cat([target_exists,
                torch.ones(len(extra), dtype=target_exists.dtype, device=device)])
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
            config=config,
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
    if bool(getattr(config, "enable_opportunist", False)):
        score = _apply_opportunist_boost(
            score=score, obs=obs, target_idx=target_idx, cand_tgt_short=cand_tgt_short,
            prod=prod, player_count=player_count, config=config,
        )
    if bool(getattr(config, "enable_weiwei", False)):
        score = _apply_weiwei_boost(
            score=score, obs=obs, obs_tensors=obs_tensors, target_idx=target_idx,
            cand_tgt_short=cand_tgt_short, config=config,
        )
    if bool(getattr(config, "enable_chenghuo", False)):
        score = _apply_chenghuo_boost(
            score=score, obs=obs, garrison_status=garrison_status, target_idx=target_idx,
            cand_tgt_short=cand_tgt_short, prod=prod, player_count=player_count, config=config,
        )
    if bool(getattr(config, "enable_qinwang", False)):
        score = _apply_qinwang_boost(
            score=score, obs=obs, target_idx=target_idx, cand_tgt_short=cand_tgt_short,
            prod=prod, player_count=player_count, config=config,
        )
    if bool(getattr(config, "enable_vulture", False)):
        allowed = _vulture_allowed_enemy_mask(obs, obs_tensors, garrison_status, prod, player_count, config)
        score = _apply_vulture_filter(score=score, obs=obs, target_idx=target_idx,
                                      cand_tgt_short=cand_tgt_short, allowed=allowed,
                                      penalty=float(getattr(config, "vulture_penalty", 0.0)))

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


def _apply_opportunist_boost(*, score, obs, target_idx, cand_tgt_short, prod, player_count, config):
    """欲擒故纵 (多人感知): catch ANY opponent at least as strong as me when it
    over-extends — boost capturing such opponents' low-garrison planets. Considers
    all 3 opponents' strength, not just the single leader."""
    P = int(obs.P); pid = int(obs.player_id)
    s = _owner_strength(obs, prod, int(player_count))
    my = float(s[pid].item()) if 0 <= pid < int(player_count) else 0.0
    strong = [o for o in range(int(player_count)) if o != pid and float(s[o].item()) >= my * 0.8]
    if not strong:
        ld = _leader_id(obs, prod, player_count, pid)
        if ld < 0:
            return score
        strong = [ld]
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs[tgt_abs]
    tgt_ships = obs.ships.to(score.dtype)[tgt_abs]
    is_strong = torch.zeros_like(tgt_owner, dtype=torch.bool)
    for o in strong:
        is_strong = is_strong | (tgt_owner == float(o))
    weak = is_strong & (tgt_ships <= float(config.opportunist_max_garrison))
    return torch.where(weak.reshape(score.shape), score * float(config.opportunist_boost), score)


def _leader_id(obs, prod, player_count, pid):
    """Strongest opponent (max prod+2.5%ships), or -1 if none alive."""
    s = _owner_strength(obs, prod, int(player_count))
    best = -1; bestv = -1.0
    for o in range(int(player_count)):
        if o == int(pid):
            continue
        v = float(s[o].item())
        if v > bestv:
            bestv = v; best = o
    return best


def _vulture_allowed_enemy_mask(obs, obs_tensors, garrison_status, prod, player_count, config):
    """[P] bool: enemy planets the 4P vulture is ALLOWED to attack — the union of
    the 四计 triggers. Everything else enemy is off-limits (不战底盘)."""
    P = int(obs.P); pid = int(obs.player_id); device = obs.ships.device; dt = obs.ships.dtype
    allowed = torch.zeros(P, dtype=torch.bool, device=device)
    enemy = (obs.owner_abs >= 0) & (obs.owner_abs != float(pid)) & obs.alive
    s = _owner_strength(obs, prod, int(player_count))
    my = float(s[pid].item()) if 0 <= pid < int(player_count) else 0.0
    leader = _leader_id(obs, prod, player_count, pid)
    # 围魏救赵: drained enemy source planets (in-flight enemy fleet from_planet)
    f = obs_tensors.get("fleets")
    if f is not None and f.numel() > 0:
        f = f.reshape(-1, f.shape[-1]); vm = (f[:, 6] > 0.5) & (f[:, 1].long() != pid) & (f[:, 1].long() >= 0)
        if bool(vm.any()):
            src = torch.zeros(P, dtype=torch.bool, device=device); src[f[:, 5].long().clamp(0, P - 1)[vm]] = True
            allowed |= enemy & src
    # 趁火打劫: enemy planet receiving fleets from a THIRD party (not me, not its owner)
    abo = getattr(garrison_status, "arrivals_by_owner", None)
    if abo is not None:
        arr = abo.sum(dim=1)
        for o in range(int(abo.shape[-1])):
            if o == pid:
                continue
            allowed |= enemy & (arr[:, o] > 0) & (obs.owner_abs != float(o))
    # 欲擒故纵: low-garrison planets of opponents at least as strong as me
    strong = [o for o in range(int(player_count)) if o != pid and float(s[o].item()) >= my * 0.8]
    if strong:
        sm = torch.zeros(P, dtype=torch.bool, device=device)
        for o in strong:
            sm |= (obs.owner_abs == float(o))
        allowed |= enemy & sm & (obs.ships.to(dt) <= float(config.opportunist_max_garrison))
    # 擒贼擒王: leader's high-production planets
    if leader >= 0:
        allowed |= (obs.owner_abs == float(leader)) & (prod.to(dt) >= float(config.qinwang_min_prod))
    return allowed


def _apply_vulture_filter(*, score, obs, target_idx, cand_tgt_short, allowed, penalty=0.0):
    """Suppress candidates attacking an ENEMY planet NOT in the 四计 allowed set.
    penalty=0 -> -inf (full pacifist block); penalty>0 -> multiply score by it
    (passive base: still contests, but defers marginal enemy attacks to 四计).
    Neutral captures + own reinforcement always pass through."""
    P = int(obs.P); pid = int(obs.player_id)
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs[tgt_abs]
    block = ((tgt_owner >= 0) & (tgt_owner != float(pid)) & (~allowed[tgt_abs])).reshape(score.shape)
    if float(penalty) <= 0.0:
        return torch.where(block, torch.full_like(score, float("-inf")), score)
    # soft: only penalize positive scores (don't lift -inf or worsen negatives oddly)
    pen = torch.where(score > 0, score * float(penalty), score)
    return torch.where(block, pen, score)


def _apply_qinwang_boost(*, score, obs, target_idx, cand_tgt_short, prod, player_count, config):
    """擒贼擒王 / 釜底抽薪 (leader-aware): focus offense on the STRONGEST opponent's
    high-production planets — knock down the real threat & starve its economy,
    not a random high-prod planet (which may belong to a dying weak player)."""
    P = int(obs.P); pid = int(obs.player_id)
    leader = _leader_id(obs, prod, player_count, pid)
    if leader < 0:
        return score
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    tgt_owner = obs.owner_abs[tgt_abs]
    tgt_prod = prod.to(score.dtype)[tgt_abs]
    king = (tgt_owner == float(leader)) & (tgt_prod >= float(config.qinwang_min_prod))
    return torch.where(king.reshape(score.shape), score * float(config.qinwang_boost), score)


def _apply_weiwei_boost(*, score, obs, obs_tensors, target_idx, cand_tgt_short, config):
    """围魏救赵 (计1): boost capturing ENEMY planets that just launched a fleet —
    their source (from_planet_id of an in-flight enemy fleet) is now drained."""
    P = int(obs.P); pid = int(obs.player_id); device = score.device
    f = obs_tensors.get("fleets")
    src_mask = torch.zeros(P, dtype=torch.bool, device=device)
    if f is not None and f.numel() > 0:
        f = f.reshape(-1, f.shape[-1])
        valid = f[:, 6] > 0.5                       # ships > 0 (skip padding)
        owner = f[:, 1].long(); frm = f[:, 5].long().clamp(0, P - 1)
        enemy = valid & (owner != pid) & (owner >= 0)
        if bool(enemy.any()):
            src_mask[frm[enemy]] = True
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    hit = src_mask[tgt_abs] & (obs.owner_abs[tgt_abs] >= 0) & (obs.owner_abs[tgt_abs] != float(pid))
    return torch.where(hit.reshape(score.shape), score * float(config.weiwei_boost), score)


def _apply_chenghuo_boost(*, score, obs, garrison_status, target_idx, cand_tgt_short, prod, player_count, config):
    """趁火打劫 (真·多人感知): boost capturing an enemy planet that a THIRD party
    (another opponent — not me, not its owner) is attacking. Uses
    arrivals_by_owner [P, K, owners] to see genuine cross-opponent fire, so we
    swoop only on what OTHERS are softening (not what I'm already taking)."""
    P = int(obs.P); pid = int(obs.player_id)
    abo = getattr(garrison_status, "arrivals_by_owner", None)
    burning = torch.zeros(P, dtype=torch.bool, device=score.device)
    if abo is not None:
        arr = abo.sum(dim=1)                              # [P, owners] total arrivals per owner
        owner_abs = obs.owner_abs
        for o in range(int(abo.shape[-1])):
            if o == pid:
                continue
            burning = burning | ((arr[:, o] > 0) & (owner_abs != float(o)) &
                                 (owner_abs >= 0) & (owner_abs != float(pid)) & obs.alive)
    if not bool(burning.any()):
        return score
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    return torch.where(burning[tgt_abs].reshape(score.shape), score * float(config.chenghuo_boost), score)


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


def _plan_evacuation(*, movement, obs, obs_tensors, cache, config, protected=None):
    """Evacuate doomed planets. A planet projected to FALL ships its garrison out
    before it falls. With 1+2 联动 (evac_counterstrike) the garrison PREFERS to
    retreat INTO a capture of the attacker's now-drained source planet (围魏救赵 =
    retreat is counter-attack); otherwise it flees to the nearest SAFE friendly
    planet (金蝉脱壳)."""
    P = int(obs.P); device = obs.device; dtype = obs.ships.dtype; pid = int(obs.player_id)
    if P == 0:
        return _empty_entries(device, dtype)
    owned = obs.owned & obs.alive
    if not bool(owned.any()):
        return _empty_entries(device, dtype)
    Hreq = int(config.evac_horizon)
    H = min(Hreq, int(movement.garrison_status(max_horizon=Hreq).ships.shape[-1]) - 1)
    if H <= 0:
        return _empty_entries(device, dtype)
    status = movement.garrison_status(max_horizon=H)
    # doom = owner flips away from me at some future step. (garrison_status.ships
    # is ABSOLUTE >=0 so the old ships<0 test NEVER fired — use owner.)
    if getattr(status, "owner", None) is not None:
        falls = status.owner[:, 1:] != pid                      # [P, H] owner-flip mask
    else:
        falls = status.ships[:, 1:] <= 0.5
    will_fall = owned & falls.any(dim=-1)
    fall_step = falls.float().argmax(dim=-1) + 1                 # first fall step per planet [P]
    # REINFORCEMENT-AWARE doom: a planet falling at step fs is "truly doomed" only
    # if my garrison + reachable rear reinforcement (by fs) still can't match the
    # incoming enemy mass. If reinforcement CAN hold it, leave it to the producer's
    # defense (don't abandon savable planets). Evac only the unsavable — and with
    # fs turns of lead time, fleeing to a safe planet is actually feasible.
    d0 = cache.cross_dist[0].to(dtype)                          # [P, P] current centre dist
    cand = will_fall & (obs.ships >= float(config.evac_min_ships))
    if protected is not None:
        cand = cand & ~protected
    ships_f = obs.ships.to(dtype)
    abo = getattr(status, "arrivals_by_owner", None)
    doomed = torch.zeros(P, dtype=torch.bool, device=device)
    for p in cand.nonzero(as_tuple=False).squeeze(1).tolist():
        fs = int(fall_step[p].item())
        if abo is not None:
            inc = float(abo[p, 1:fs + 1, :].sum().item() - abo[p, 1:fs + 1, pid].sum().item())
        else:
            inc = float(ships_f[p].item()) + 1.0
        others = owned.clone(); others[p] = False
        oi = others.nonzero(as_tuple=False).squeeze(1)
        reinf = 0.0
        if oi.numel() > 0:
            sp = fleet_speed(ships_f[oi].clamp(min=1.0))
            eta_r = (d0[oi, p] / sp.clamp(min=1e-6)).ceil()
            reinf = float(ships_f[oi][eta_r < float(fs)].sum().item())
        if float(ships_f[p].item()) + reinf < inc:              # unsavable even with reinforcement
            doomed[p] = True
    safe = owned & ~will_fall                                    # owned planets that hold
    if not bool(doomed.any()) or (not bool(safe.any()) and not bool(config.evac_counterstrike)):
        return _empty_entries(device, dtype)
    src_idx = doomed.nonzero(as_tuple=False).squeeze(1)
    safe_idx = safe.nonzero(as_tuple=False).squeeze(1)

    # 围魏 targets: drained enemy SOURCE planets (from_planet_id of in-flight enemy fleets)
    counter_mask = torch.zeros(P, dtype=torch.bool, device=device)
    if bool(config.evac_counterstrike):
        f = obs_tensors.get("fleets")
        if f is not None and f.numel() > 0:
            f = f.reshape(-1, f.shape[-1])
            vmask = (f[:, 6] > 0.5) & (f[:, 1].long() != pid) & (f[:, 1].long() >= 0)
            if bool(vmask.any()):
                counter_mask[f[:, 5].long().clamp(0, P - 1)[vmask]] = True
        counter_mask &= (obs.owner_abs >= 0) & (obs.owner_abs != float(pid)) & obs.alive
    counter_idx = counter_mask.nonzero(as_tuple=False).squeeze(1) if bool(counter_mask.any()) else None

    ev_src = []; ev_dst = []; ev_ships = []
    for s_i in range(int(src_idx.shape[0])):
        if len(ev_src) >= int(config.evac_max_waves):
            break
        s = int(src_idx[s_i].item())
        gs = float(obs.ships[s].item())
        fs = int(fall_step[s].item())
        speed = float(fleet_speed(torch.tensor(max(gs, 1.0), dtype=dtype, device=device)))
        dst = None
        # --- 1+2 联动: prefer a winnable counterstrike on the attacker's drained source ---
        if counter_idx is not None and counter_idx.numel() > 0:
            cd = d0[s, counter_idx]
            ceta = (cd / max(speed, 1e-6)).ceil()
            win = (gs > obs.ships.to(dtype)[counter_idx] * float(config.evac_counter_margin)) & (ceta <= float(H)) & (counter_idx != s)
            if bool(win.any()):
                cb = int(torch.where(win, cd, torch.full_like(cd, 1e9)).argmin().item())
                dst = int(counter_idx[cb].item())
        # --- fallback: 金蝉脱壳 flee to nearest safe friendly reachable before fall ---
        if dst is None and safe_idx.numel() > 0:
            dists = d0[s, safe_idx]
            etas = (dists / max(speed, 1e-6)).ceil()
            valid = (etas < float(fs + 1)) & (safe_idx != s)
            if bool(valid.any()):
                best = int(torch.where(valid, dists, torch.full_like(dists, 1e9)).argmin().item())
                dst = int(safe_idx[best].item())
        if dst is None:
            continue
        ev_src.append(s); ev_dst.append(dst); ev_ships.append(gs)

    if not ev_src:
        return _empty_entries(device, dtype)
    # build a proper LaunchEntries (with angle, like wave/regroup entries) — using
    # make_launch_set's LaunchSet (no angle) crashes the downstream concat/infer.
    src_t = torch.tensor(ev_src, dtype=torch.long, device=device)
    dst_t = torch.tensor(ev_dst, dtype=torch.long, device=device)
    ships_t = torch.tensor(ev_ships, dtype=dtype, device=device)
    aim = intercept_angle(movement, src_t, dst_t, ships_t)
    return LaunchEntries(
        source_slots=src_t, target_slots=dst_t, ships=ships_t,
        angle=aim["angle"].reshape(-1), eta=aim["eta"].reshape(-1),
        valid=torch.ones(len(ev_src), dtype=torch.bool, device=device),
    )


def _predict_opponent_arrivals(movement, obs, config, player_count: int):
    """Each opponent's strongest ``react_sources_per_opp`` planets launch their
    whole garrison at the nearest *takeable* target (alive, not theirs, current
    defenders < their garrison). Returns (target_slots, owners, ships, eta,
    valid) or None. (ported from single-size/main_react.py)"""
    P = obs.P
    device = obs.device
    dt = obs.ships.dtype
    pid = int(obs.player_id)
    cur = obs.ships.to(dt)
    n_src = max(1, int(config.react_sources_per_opp))

    src_slots, tgt_slots, owners, sizes = [], [], [], []
    for o in range(int(player_count)):
        if o == pid:
            continue
        o_mask = obs.alive & (obs.owner_abs == float(o)) & (cur >= float(config.min_ships_to_launch))
        if not bool(o_mask.any()):
            continue
        o_idx, o_exists = _candidate_indices(cur, o_mask, n_src)
        for i in range(int(o_idx.shape[0])):
            if not bool(o_exists[i]):
                continue
            s = int(o_idx[i])
            ss = float(cur[s])
            takeable = obs.alive & (obs.owner_abs != float(o)) & (cur < ss)
            takeable[s] = False
            if not bool(takeable.any()):
                continue
            d2 = (obs.x - obs.x[s]) ** 2 + (obs.y - obs.y[s]) ** 2
            d2 = torch.where(takeable, d2, torch.full_like(d2, float("inf")))
            t = int(torch.argmin(d2))
            src_slots.append(s); tgt_slots.append(t); owners.append(o); sizes.append(ss)

    if not src_slots:
        return None
    src = torch.tensor(src_slots, dtype=torch.long, device=device)
    tgt = torch.tensor(tgt_slots, dtype=torch.long, device=device)
    own = torch.tensor(owners, dtype=torch.long, device=device)
    shp = torch.tensor(sizes, dtype=dt, device=device)
    aim = intercept_angle(movement, src, tgt, shp)
    return tgt, own, shp, aim["eta"], aim["viable"]


def _reactive_status(movement, obs, config, player_count: int, H: int):
    """garrison_status with predicted opponent arrivals injected, then fully
    reverted so the rolling fleet cache is unaffected next turn."""
    pred = _predict_opponent_arrivals(movement, obs, config, player_count)
    if pred is None:
        return movement.garrison_status(max_horizon=H)
    tgt, own, shp, eta, valid = pred
    snap = {
        "fb": None if movement.fleet_buckets is None else movement.fleet_buckets.clone(),
        "go": None if movement.garrison_owner_cache is None else movement.garrison_owner_cache.clone(),
        "gs": None if movement.garrison_ships_cache is None else movement.garrison_ships_cache.clone(),
        "pco": None if movement.garrison_pre_combat_owner_cache is None else movement.garrison_pre_combat_owner_cache.clone(),
        "pcs": None if movement.garrison_pre_combat_ships_cache is None else movement.garrison_pre_combat_ships_cache.clone(),
        "df": None if movement.garrison_dirty_from is None else movement.garrison_dirty_from.clone(),
    }
    try:
        movement.record_fleet_arrivals(target_slots=tgt, owner_ids=own, ships=shp, eta=eta, valid=valid)
        status = movement.garrison_status(max_horizon=H)
        status = type(status)(
            owner=status.owner.clone(), ships=status.ships.clone(),
            pre_combat_owner=None if status.pre_combat_owner is None else status.pre_combat_owner.clone(),
            pre_combat_ships=None if status.pre_combat_ships is None else status.pre_combat_ships.clone(),
            arrivals_by_owner=None if status.arrivals_by_owner is None else status.arrivals_by_owner.clone(),
        )
    finally:
        movement.fleet_buckets = snap["fb"]
        movement.garrison_owner_cache = snap["go"]
        movement.garrison_ships_cache = snap["gs"]
        movement.garrison_pre_combat_owner_cache = snap["pco"]
        movement.garrison_pre_combat_ships_cache = snap["pcs"]
        movement.garrison_dirty_from = snap["df"]
    return status


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
    if bool(getattr(config, "react_opponent", False)):
        status = _reactive_status(movement, obs, config, int(player_count), H)
    else:
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
    if bool(getattr(config, "enable_evacuation", False)):
        # planets the producer is already sending ships to (contesting/defending)
        protected = torch.zeros(int(obs.P), dtype=torch.bool, device=device)
        vmask = getattr(entries, "valid", None)
        for fld in ("target_slots", "source_slots"):   # exclude planets I send TO or FROM
            sl = getattr(entries, fld, None)
            if sl is not None and sl.numel() > 0:
                sel = sl[vmask] if (vmask is not None and vmask.shape == sl.shape) else sl.reshape(-1)
                if sel.numel() > 0:
                    protected[sel.reshape(-1).clamp(0, int(obs.P) - 1)] = True
        evac_entries = _plan_evacuation(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, config=config, protected=protected)
        # evac first = priority: doomed garrisons leave before being double-spent
        entries = concat_launch_entries([evac_entries, entries])
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
# 分治路由 2P engine: agent_main base + beta3.0 (本地最强 2P, lightver 83)
CONFIG_2P = _ow_apply_config_overrides(dataclasses.replace(ProducerLiteConfig(), reinforce_size_beta=3.0), _OW_STRATEGY_PARAMS.get("config_2p"))
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
