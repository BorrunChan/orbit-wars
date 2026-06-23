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
    hoard_min_planets: int = 15      # 硬编码生效 (hoard_g2 策略): 拥有>=15星后屯兵
    hoard_roi_mult: float = 2.0      # 屯兵时 roi×2 (抬高发兵门槛, 堆生产)
    hoard_max_waves: int = 0         # cap waves when hoarding (0 = unchanged)
    enable_comet: bool = True        # 彗星跳板, 硬编码生效
    comet_prod_bonus: float = 4.0
    comet_clear_remaining: int = 7   # 彗星剩余存活<=N步时清仓全力进攻(离场前把驻军发出去)
    enable_comet_value: bool = True  # 用 path 预测轨迹目标, 只占"会划到肥目标"的彗星(止损没价值的)
    comet_reach: float = 30.0        # 彗星轨迹点到目标的可达距离(够近才算够得到打)
    comet_value_window: int = 18     # 看未来多少步轨迹
    comet_value_threshold: float = 2.0  # 轨迹可达目标的加权价值(prod/守军) >= 此值才占
    comet_attack_min_prod: float = 0.0  # V7 去保守: 0=不限制彗星打敌(V6克制版Kaggle暴跌961, 进攻保守是毒药)
    enable_opportunist: bool = True   # 欲擒: 打弱守强敌(彗星投送敌后配此最佳) — 连招默认开
    opportunist_boost: float = 1.5
    opportunist_max_garrison: float = 15.0
    enable_chenghuo: bool = True      # 趁火: 打被第三方攻击的敌星 — 连招默认开
    chenghuo_boost: float = 1.5
    # --- 僵局破局: 学 Kaggle/Xander 主流多路并发打法 (75-86% 对局达 8+ 并发舰) ----
    # 中立扩张饱和 + 我方扩张停滞 = 屯兵僵局(Xander 局根因). 此时 roi 门槛让"打敌星"
    # 的兵发不出去 → 静态挨打. 触发后降 roi(边际攻击也发) + 提 waves(多路并发) 主动
    # 打敌星薄弱点破局. 只动 roi/waves: capture_floor + safe_drain 仍保证不发送死兵.
    enable_stalemate_break: bool = True
    stalemate_min_step: int = 70       # 过早期扩张后才考虑 (早期照常扩张)
    stalemate_stall_turns: int = 15    # 连续 N 步 owned 无新增 = 扩张停滞/僵局
    stalemate_min_planets: int = 4     # 有产能基础(多源)才多路, 否则保守扩张
    stalemate_roi: float = 1.05        # 破局降发兵门槛 (僵局打敌星 roi 难过默认 1.5)
    stalemate_waves: int = 10          # 破局多路并发波数 (默认 7)
    capture_overhead: float = 1.0      # 攻星安全系数 (1.0=刚好够, <1=冒险试探打不下也发)
    stalemate_capture_overhead: float = 0.9  # 破局放宽: 真·多路试探(差一点也发,学Xander突破)
    stalemate_regroup_delta: float = 0.15    # 破局降回防门槛 (小威胁也回防, 抗多路并发)
    # --- 出逃保兵 (金蝉脱壳 + 围魏救赵): 被集火打崩的星, 撤全兵保实力 + 反打攻击者 ----
    # 已抽干的源星(它把兵都发来打我了, 正好反打它空虚的本星). 留得青山在不怕没柴烧.
    # 支线A 实证 4P +6(唯一正收益旋钮). 只撤"连后方增援都救不了"的真·守不住星.
    enable_evacuation: bool = False
    evac_min_ships: float = 10.0
    evac_horizon: int = 10
    evac_max_waves: int = 4
    evac_counterstrike: bool = True
    evac_counter_margin: float = 1.25


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
    src_list = src_mask.nonzero(as_tuple=False).squeeze(1)
    Ns = int(src_list.numel()); Nt = int(tgt_idx.numel())
    # 太阳/碰撞规避: 预算每个源(全garrison)到所有目标的 viable(路径不穿太阳/星体).
    # intercept_angle 的 viable 复现引擎首次接触判定 → False = 中途撞太阳/越界/撞星,会全军覆没.
    src_ships = obs.ships.to(dtype)[src_list]
    sizes_t = src_ships.view(Ns, 1).expand(Ns, Nt)
    aimT = intercept_angle(movement, src_list.view(Ns, 1), tgt_idx.view(1, Nt), sizes_t)
    viableT = aimT["viable"].reshape(Ns, Nt)
    fr_idx = friend.nonzero(as_tuple=False).squeeze(1)
    Nf = int(fr_idx.numel())
    if Nf > 0:
        sizes_f = src_ships.view(Ns, 1).expand(Ns, Nf)
        aimF = intercept_angle(movement, src_list.view(Ns, 1), fr_idx.view(1, Nf), sizes_f)
        viableF = aimF["viable"].reshape(Ns, Nf)
    ev_s = []; ev_d = []; ev_n = []
    for row, s in enumerate(src_list.tolist()):
        n = float(int(obs.ships[s].item()))                # 全 garrison, 整数
        if n < float(config.min_ships_to_launch):
            continue
        # 向高分看齐(>1200 仅18%打敌76%回援): 只打"能打下+不撞太阳+prod够肥(≥门槛)"的敌星;
        # 瘦敌星(prod<门槛, 占我方彗星打敌大头)不打 → 落入下方撤友星回援保兵(兵力调度而非强袭瘦敌)
        winnable = (tgt_ships * 1.1 < n) & (tgt_idx != s) & viableT[row] & (tgt_prod >= float(config.comet_attack_min_prod))
        if bool(winnable.any()):
            cand = tgt_idx[winnable]; cand_prod = tgt_prod[winnable]
            best = int(cand[int(cand_prod.argmax().item())].item())
        elif Nf > 0:
            frow = viableF[row] & (fr_idx != s)
            if not bool(frow.any()):
                continue
            cand_fi = fr_idx[frow]
            best = int(cand_fi[int(d0[s, cand_fi].argmin().item())].item())
        else:
            continue
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
        valid=aim["viable"].reshape(-1),   # 用引擎 viable 而非 ones — 不发撞太阳的清仓
    )


def _owner_strength(obs, prod, player_count):
    device = obs.device; dtype = obs.ships.dtype
    strength = torch.zeros(int(player_count), dtype=dtype, device=device)
    owner = obs.owner_abs.to(torch.long); prod_v = prod.to(dtype); ships = obs.ships.to(dtype)
    for oid in range(int(player_count)):
        mask = obs.alive & (owner == oid)
        if bool(mask.any()):
            strength[oid] = prod_v[mask].sum() + 0.025 * ships[mask].sum()
    return strength


def _leader_id(obs, prod, player_count, pid):
    s = _owner_strength(obs, prod, int(player_count))
    best = -1; bestv = -1.0
    for o in range(int(player_count)):
        if o == int(pid):
            continue
        v = float(s[o].item())
        if v > bestv:
            bestv = v; best = o
    return best


def _apply_opportunist_boost(*, score, obs, target_idx, cand_tgt_short, prod, player_count, config):
    """欲擒故纵: 打任何"强度>=我"的对手的低守军星(它过度扩张时的弱点). 彗星投送敌后配此."""
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
    tgt_owner = obs.owner_abs[tgt_abs]; tgt_ships = obs.ships.to(score.dtype)[tgt_abs]
    is_strong = torch.zeros_like(tgt_owner, dtype=torch.bool)
    for o in strong:
        is_strong = is_strong | (tgt_owner == float(o))
    weak = is_strong & (tgt_ships <= float(config.opportunist_max_garrison))
    return torch.where(weak.reshape(score.shape), score * float(config.opportunist_boost), score)


def _apply_chenghuo_boost(*, score, obs, garrison_status, target_idx, cand_tgt_short, prod, player_count, config):
    """趁火打劫: 打"被第三方(别的对手)正在攻击"的敌星(arrivals_by_owner 看跨对手火力)."""
    P = int(obs.P); pid = int(obs.player_id)
    abo = getattr(garrison_status, "arrivals_by_owner", None)
    burning = torch.zeros(P, dtype=torch.bool, device=score.device)
    if abo is not None:
        arr = abo.sum(dim=1); owner_abs = obs.owner_abs
        for o in range(int(abo.shape[-1])):
            if o == pid:
                continue
            burning = burning | ((arr[:, o] > 0) & (owner_abs != float(o)) &
                                 (owner_abs >= 0) & (owner_abs != float(pid)) & obs.alive)
    if not bool(burning.any()):
        return score
    tgt_abs = target_idx[cand_tgt_short].clamp(0, P - 1)
    return torch.where(burning[tgt_abs].reshape(score.shape), score * float(config.chenghuo_boost), score)


def _comet_worth_mask(obs, obs_tensors, prod, config):
    """用彗星 path 预测未来轨迹, 返回 [P] bool: 哪些彗星值得占(未来会划到肥的敌/中立
    目标附近). 目标移动忽略(短期用当前位置近似). 这是彗星价值评估的地基: 止损那些
    轨迹全是空域/我方区的没价值彗星(占了反而拖累)."""
    P = int(obs.P); device = obs.device; dtype = obs.ships.dtype
    worth = torch.zeros(P, dtype=torch.bool, device=device)
    comets = obs_tensors.get("comets")
    if not isinstance(comets, dict):
        return worth
    pidx = comets.get("path_index"); paths = comets.get("paths"); cpids = comets.get("planet_ids")
    if pidx is None or paths is None or cpids is None:
        return worth
    planets = obs_tensors["planets"]
    pl_ids = planets[..., 0].long().reshape(-1)
    px = planets[..., 2].reshape(-1).to(dtype); py = planets[..., 3].reshape(-1).to(dtype)
    tgt = obs.alive & (~obs.owned)                          # 敌/中立目标
    tgt_idx = tgt.nonzero(as_tuple=False).squeeze(1)
    if tgt_idx.numel() == 0:
        return worth
    tx = px[tgt_idx]; ty = py[tgt_idx]
    tprod = prod.reshape(-1).to(dtype)[tgt_idx]
    tships = obs.ships.reshape(-1).to(dtype)[tgt_idx]
    reach = float(config.comet_reach); window = int(config.comet_value_window)
    thr = float(config.comet_value_threshold)
    for e in range(int(pidx.shape[0])):
        pe = int(pidx[e].item())
        if pe < 0:
            continue
        for c in range(int(paths.shape[1])):
            cid = int(cpids[e, c].item())
            if cid < 0:
                continue
            sel = (pl_ids == cid).nonzero(as_tuple=False)
            if sel.numel() == 0:
                continue
            ci = int(sel[0].item())
            if bool(obs.owned[ci]):
                continue
            fut = paths[e, c, pe:pe + window].to(dtype)     # 未来轨迹点 [W,2]
            if fut.shape[0] == 0:
                continue
            dx = fut[:, 0].unsqueeze(1) - tx.unsqueeze(0)   # [W,T]
            dy = fut[:, 1].unsqueeze(1) - ty.unsqueeze(0)
            mind = (dx * dx + dy * dy).sqrt().min(dim=0).values  # [T] 目标到轨迹最近距
            reachable = mind < reach
            if bool(reachable.any()):
                val = float((tprod[reachable] / (1.0 + 0.1 * tships[reachable])).sum().item())
                if val >= thr:
                    worth[ci] = True
    return worth


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
                if bool(getattr(config, "enable_comet_value", False)):
                    cm = cm & _comet_worth_mask(obs, obs_tensors, prod, config)  # 只占值得的彗星
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
        capture_overhead=float(getattr(config, "capture_overhead", 1.0)), player_id=pid, reinforcement=reinforcement,
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

    # 三计 (彗星把兵投送到敌后, 在那选弱点打): 欲擒打弱守强敌, 趁火打被第三方攻击的敌星
    if bool(getattr(config, "enable_opportunist", False)):
        score = _apply_opportunist_boost(score=score, obs=obs, target_idx=target_idx,
            cand_tgt_short=cand_tgt_short, prod=prod, player_count=player_count, config=config)
    if bool(getattr(config, "enable_chenghuo", False)):
        score = _apply_chenghuo_boost(score=score, obs=obs, garrison_status=garrison_status,
            target_idx=target_idx, cand_tgt_short=cand_tgt_short, prod=prod,
            player_count=player_count, config=config)

    # 向高分看齐(>1200 彗星仅18%打敌76%回援): 彗星源星不打瘦敌(prod<门槛) → 该候选 -inf,
    # 兵落入 leftover 由 regroup 回援/巩固自己 (彗星当兵力调度跳板, 不强袭守得住的瘦敌)
    _camp = float(getattr(config, "comet_attack_min_prod", 0.0))
    if _camp > 0:
        _cids = obs_tensors.get("comet_planet_ids")
        if _cids is not None and _cids.numel() > 0:
            _cset = _cids.reshape(-1); _cset = _cset[_cset >= 0]
            if _cset.numel() > 0:
                _plids = obs_tensors["planets"][..., 0].long().reshape(-1)
                _srcabs = cand_src.reshape(-1).clamp(0, P - 1)
                _src_is_comet = (_plids[_srcabs].unsqueeze(1) == _cset.long().unsqueeze(0)).any(dim=1)
                _tgtabs = target_idx[cand_tgt_short].clamp(0, P - 1)
                _tgt_enemy = (obs.owner_abs[_tgtabs] >= 0) & (obs.owner_abs[_tgtabs] != float(pid))
                _tgt_thin = prod[_tgtabs] < _camp
                _suppress = _src_is_comet & _tgt_enemy & _tgt_thin
                score = torch.where(_suppress, torch.full_like(score, float("-inf")), score)

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


def _apply_stalemate_config(config: ProducerLiteConfig, obs, *, step: int, player_count: int, memory) -> ProducerLiteConfig:
    """中立扩张饱和 + 扩张停滞的僵局 → 切多路并发打敌星 (降 roi + 提 waves).

    扩张停滞用 memory 跟踪: owned 创新高则清零停滞计数, 否则 +1. 连续 stall_turns 步无
    新增星 = 僵局. 学 Kaggle 主流: 与其屯兵静态挨打, 不如主动多路压制敌星薄弱点破局.
    只在 2P (4P prod-only 且难屯到僵局, 另路由 default). 关 hoard 抬价以免互相抵消."""
    if not bool(getattr(config, "enable_stalemate_break", False)) or int(player_count) >= 4:
        return config
    owned = int(obs.owned.sum().item())
    if owned > int(getattr(memory, "peak_owned", 0)):
        memory.peak_owned = owned
        memory.stall_steps = 0
    else:
        memory.stall_steps = int(getattr(memory, "stall_steps", 0)) + 1
    if int(step) < int(config.stalemate_min_step) or owned < int(config.stalemate_min_planets):
        return config
    if int(getattr(memory, "stall_steps", 0)) < int(config.stalemate_stall_turns):
        return config
    enemy = obs.alive & (~obs.owned) & (obs.owner_abs >= 0)   # 有敌方星才值得破局
    if not bool(enemy.any()):
        return config
    return dataclasses.replace(
        config,
        roi_threshold=float(config.stalemate_roi),
        max_waves_per_turn=max(int(config.max_waves_per_turn), int(config.stalemate_waves)),
        hoard_min_planets=0,   # 破局时关 hoard, 别再抬 roi 屯兵
        capture_overhead=float(config.stalemate_capture_overhead),  # 进攻: 放宽试探多路突破
        regroup_pressure_delta_min=float(config.stalemate_regroup_delta),  # 防守: 小威胁也回防
    )


def _plan_evacuation(*, movement, obs, obs_tensors, cache, config, protected=None):
    """守不住的星出逃保兵(金蝉脱壳)+ 优先反打攻击者已抽干的源星(围魏救赵).
    被多家集火打崩时撤全兵保实力, 留得青山在不怕没柴烧. 只撤"连后方增援都救不了"的真·守不住星."""
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
    if getattr(status, "owner", None) is not None:
        falls = status.owner[:, 1:] != pid                          # owner-flip = 将易主
    else:
        falls = status.ships[:, 1:] <= 0.5
    will_fall = owned & falls.any(dim=-1)
    fall_step = falls.float().argmax(dim=-1) + 1
    d0 = cache.cross_dist[0].to(dtype)
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
        if float(ships_f[p].item()) + reinf < inc:                  # 增援也救不了 = 真·守不住
            doomed[p] = True
    safe = owned & ~will_fall
    if not bool(doomed.any()) or (not bool(safe.any()) and not bool(config.evac_counterstrike)):
        return _empty_entries(device, dtype)
    src_idx = doomed.nonzero(as_tuple=False).squeeze(1)
    safe_idx = safe.nonzero(as_tuple=False).squeeze(1)
    # 围魏 targets: 攻击者已抽干的源星 (in-flight 敌方 fleet 的 from_planet_id)
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
        # 1+2 联动: 优先反打攻击者已抽干的源星 (撤退即反击)
        if counter_idx is not None and counter_idx.numel() > 0:
            cd = d0[s, counter_idx]
            ceta = (cd / max(speed, 1e-6)).ceil()
            win = (gs > obs.ships.to(dtype)[counter_idx] * float(config.evac_counter_margin)) & (ceta <= float(H)) & (counter_idx != s)
            if bool(win.any()):
                cb = int(torch.where(win, cd, torch.full_like(cd, 1e9)).argmin().item())
                dst = int(counter_idx[cb].item())
        # fallback: 金蝉脱壳 撤到最近且赶得及的安全友星
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
    src_t = torch.tensor(ev_src, dtype=torch.long, device=device)
    dst_t = torch.tensor(ev_dst, dtype=torch.long, device=device)
    ships_t = torch.tensor(ev_ships, dtype=dtype, device=device)
    aim = intercept_angle(movement, src_t, dst_t, ships_t)
    return LaunchEntries(
        source_slots=src_t, target_slots=dst_t, ships=ships_t,
        angle=aim["angle"].reshape(-1), eta=aim["eta"].reshape(-1),
        valid=aim["viable"].reshape(-1),   # 修撞太阳 bug: 用 viable 而非 ones (出逃也不穿太阳)
    )


def run_turn(obs_tensors: dict, *, config: ProducerLiteConfig, player_count: int, memory) -> dict:
    device = obs_tensors["planets"].device
    obs = parse_obs(obs_tensors)
    P = obs.P
    if P == 0:
        return empty_action_row(device)
    config = _apply_hoard_config(config, int(obs.owned.sum().item()))
    config = _apply_stalemate_config(
        config, obs, step=int(obs_tensors["step"].reshape(-1)[0].item()),
        player_count=int(player_count), memory=memory,
    )

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
    if bool(getattr(config, "enable_evacuation", False)):
        protected = torch.zeros(int(obs.P), dtype=torch.bool, device=device)
        vmask = getattr(entries, "valid", None)
        for fld in ("target_slots", "source_slots"):   # producer 正在发兵的星别撤
            sl = getattr(entries, fld, None)
            if sl is not None and sl.numel() > 0:
                sel = sl[vmask] if (vmask is not None and vmask.shape == sl.shape) else sl.reshape(-1)
                if sel.numel() > 0:
                    protected[sel.reshape(-1).clamp(0, int(obs.P) - 1)] = True
        evac_entries = _plan_evacuation(movement=movement, obs=obs, obs_tensors=obs_tensors, cache=cache, config=config, protected=protected)
        entries = concat_launch_entries([evac_entries, entries])   # 出逃优先: 守不住的兵先撤
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
    reinforce_size_beta=1.5,  # V7 适度激进打敌 (beta2.2→1.5, 本地4P夺冠30→38; 去保守回V5+激进). evac/克制已去(V6保守化Kaggle暴跌961)
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
# 连招 2P base: size 细分 (.33,.66,1) — 攻克 lightver 关键旋钮 (33→赢). 三计/彗星继承 ProducerLiteConfig 默认.
CONFIG_2P = dataclasses.replace(ProducerLiteConfig(), size_multipliers=(0.33, 0.66, 1.0))
CONFIG_2P = _ow_apply_config_overrides(CONFIG_2P, _OW_STRATEGY_PARAMS.get("config_2p"))
CONFIG_4P = _ow_apply_config_overrides(CONFIG_4P, _OW_STRATEGY_PARAMS.get("config_4p"))


def _config_for(player_count: int) -> ProducerLiteConfig:
    return CONFIG_4P if int(player_count) >= 4 else CONFIG_2P


class ProducerLiteMemory:
    def __init__(self) -> None:
        self.movement = None
        self.cached_player_count: int | None = None
        self.last_sparse_action_row: dict | None = None
        self.peak_owned: int = 0
        self.stall_steps: int = 0

    def reset(self) -> None:
        self.movement = None
        self.cached_player_count = None
        self.last_sparse_action_row = None
        self.peak_owned = 0
        self.stall_steps = 0


class ProducerLiteRuntime:
    def __init__(self, memory: ProducerLiteMemory | None = None) -> None:
        self.memory = memory if memory is not None else ProducerLiteMemory()

    def reset(self) -> None:
        self.memory.reset()

    def tensor_action(self, obs_tensors: dict):
        mem = self.memory
        if bool((obs_tensors["step"] == 0).all()):
            mem.cached_player_count = None
            mem.peak_owned = 0
            mem.stall_steps = 0
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
