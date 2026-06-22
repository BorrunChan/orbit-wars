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
    LaunchEntries,
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
    hoard_min_planets: int = 0      # 硬编码生效 (hoard_g2 策略): 拥有>=15星后屯兵
    hoard_roi_mult: float = 1.0      # 屯兵时 roi×2 (抬高发兵门槛, 堆生产)
    hoard_max_waves: int = 0         # cap waves when hoarding (0 = unchanged)
    enable_comet: bool = True        # 彗星跳板, 硬编码生效
    comet_prod_bonus: float = 4.0
    comet_clear_remaining: int = 7   # 彗星剩余存活<=N步时清仓全力进攻(离场前把驻军发出去)


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


def _comet_clearance(movement, obs, obs_tensors, cache, config, prod):
    """彗星离场前清仓: 占领的、剩余存活<=N步的彗星, 把全部驻军投到最近的非我星(进攻).
    彗星 step~84 会划出地图消失, 囤在上面的兵会随之白丢 → 离场前必须发出去."""
    P = int(obs.P); device = obs.device; dtype = obs.ships.dtype; pid = int(obs.player_id)
    comets = obs_tensors.get("comets")
    if not isinstance(comets, dict) or P == 0:
        return _empty_entries(device, dtype)
    pidx = comets.get("path_index"); paths = comets.get("paths"); cpids = comets.get("planet_ids")
    if pidx is None or paths is None or cpids is None:
        return _empty_entries(device, dtype)
    urgent = set()
    for e in range(int(pidx.shape[0])):
        pe = int(pidx[e].item())
        if pe < 0:
            continue
        path = paths[e, 0]                                   # [L,2] (同 event 的彗星同 path 长)
        moving = (path[1:] - path[:-1]).abs().sum(-1) > 1e-3
        plen = int(moving.sum().item()) + 1                 # 实际 path 长 (去 padding)
        if plen - pe <= int(config.comet_clear_remaining):  # 即将离场
            for c in cpids[e].tolist():
                if c >= 0:
                    urgent.add(int(c))
    if not urgent:
        return _empty_entries(device, dtype)
    pl_ids = obs_tensors["planets"][..., 0].long().reshape(-1)
    urgent_mask = torch.zeros(P, dtype=torch.bool, device=device)
    for i in range(P):
        if int(pl_ids[i].item()) in urgent:
            urgent_mask[i] = True
    src_mask = obs.owned & obs.alive & urgent_mask & (obs.ships >= float(config.min_ships_to_launch))
    tgt_mask = obs.alive & (~obs.owned)                     # 最近的非我星(敌/中立)
    if not bool(src_mask.any()) or not bool(tgt_mask.any()):
        return _empty_entries(device, dtype)
    d0 = cache.cross_dist[0].to(dtype)
    tgt_idx = tgt_mask.nonzero(as_tuple=False).squeeze(1)
    tgt_ships = obs.ships.to(dtype)[tgt_idx]
    tgt_prod = prod.to(dtype)[tgt_idx]
    friend = obs.owned & obs.alive
    ev_s = []; ev_d = []; ev_n = []
    for s in src_mask.nonzero(as_tuple=False).squeeze(1).tolist():
        n = float(int(obs.ships[s].item()))                # 全 garrison, 整数
        if n < float(config.min_ships_to_launch):
            continue
        # 优先: 全兵能打下的目标(守军*1.1<我兵)里 prod 最高(价值最大);
        # 都打不下: 撤到最近友星保兵(而非全兵送死)
        winnable = (tgt_ships * 1.1 < n) & (tgt_idx != s)
        if bool(winnable.any()):
            cand = tgt_idx[winnable]; cand_prod = tgt_prod[winnable]
            best = int(cand[int(cand_prod.argmax().item())].item())
        else:
            fr = friend.clone(); fr[s] = False
            fi = fr.nonzero(as_tuple=False).squeeze(1)
            if fi.numel() == 0:
                continue
            best = int(fi[int(d0[s, fi].argmin().item())].item())
        ev_s.append(s); ev_d.append(best); ev_n.append(n)
    if not ev_s:
        return _empty_entries(device, dtype)
    src_t = torch.tensor(ev_s, dtype=torch.long, device=device)
    dst_t = torch.tensor(ev_d, dtype=torch.long, device=device)
    ships_t = torch.tensor(ev_n, dtype=dtype, device=device)
    aim = intercept_angle(movement, src_t, dst_t, ships_t)
    return LaunchEntries(
        source_slots=src_t, target_slots=dst_t, ships=ships_t,
        angle=aim["angle"].reshape(-1), eta=aim["eta"].reshape(-1),
        valid=torch.ones(len(ev_s), dtype=torch.bool, device=device),
    )


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
    enemy_mass = (
        cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        if beta > 0.0 or bool(config.enable_regroup) else None
    )
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

    wave_entries, leftover = _greedy_select(
        P=P, W=W, device=device, dtype=dtype, score=score,
        cand_src=cand_src, cand_send=cand_send, cand_angle=cand_angle, cand_eta=cand_eta,
        cand_active=cand_active, cand_tgt_slot=cand_tgt_slot, cand_tgt_short=cand_tgt_short,
        cand_is_def=cand_is_def, source_budget=obs.ships.to(dtype).clone(),
        target_exists=target_exists, roi_threshold=float(config.roi_threshold),
    )

    if bool(config.enable_regroup):
        if enemy_mass is None:
            enemy_mass = cheap_enemy_pressure(obs, cache, horizon=float(K_eta), player_id=pid)
        regroup_entries = _plan_regroup(
            movement=movement, obs=obs, obs_tensors=obs_tensors, garrison_status=garrison_status,
            leftover=leftover, original_ships=obs.ships.to(dtype), pressure=enemy_mass,
            config=config, H=H,
        )
        out = concat_launch_entries([wave_entries, regroup_entries])
    else:
        out = wave_entries
    # 彗星离场前清仓: 占领的彗星即将消失 -> 全力进攻最近非我星 (兵不发随彗星白丢)
    if bool(getattr(config, "enable_comet", False)):
        clear = _comet_clearance(movement, obs, obs_tensors, cache, config, prod)
        if clear is not None and clear.valid is not None and bool(clear.valid.any()):
            out = concat_launch_entries([out, clear])
    return out


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
    cache = build_distance_cache(movement, max_k=int(config.horizon))
    H = int(config.horizon)
    status = movement.garrison_status(max_horizon=H)
    alive_by_step = movement.alive_by_step[: H + 1]

    entries = plan_lite_waves(
        movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache,
        garrison_status=status, prod=movement.planet_prod,
        alive_by_step=alive_by_step, config=config, player_count=int(player_count),
    )
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
