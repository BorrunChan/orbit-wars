# Orbit Wars 当前最优 agent `rmargin` —— 完整算法详解（从零）

> **命名 `rmargin`**（reinforcement margin，增援余量版）。当前 Kaggle **public ~1245**（1241–1247，最高提交 ref 53704199；代码 `external/reyhan/`，参数化副本 `search/agent_main.py`，引擎包 `orbit_lite/`）。
> 衍生 `rmargin-hoard`（+ 显式屯兵旋钮，ref 53730874）实测 = 1245 持平、无增益（见 §7）。
> 本文**自包含、从零**：游戏物理 → 状态 → 三大预计算 → 每回合决策（逐步+伪代码）→ 精确评分原理 → 2P/4P 策略不对称 → 为何强 → 复杂度/部署。不假设了解任何先前版本。

---

## 0. 一句话

每回合：对「我所有星 × 附近所有可达目标」的组合，各假设「发一支舰队过去」，用一个**引擎级精确前向模拟**算出「这一发让(我方净收益 − 对手净收益)变化多少」，**贪心**挑出最划算的几支发出；剩余兵力沿「敌方压力梯度」向前线调度。全程 torch 张量并行，每回合 ~50ms。

---

## 1. 游戏与物理（规则基础）

### 1.1 棋盘与星球
- **棋盘**：100×100 连续平面；正中 (50,50) 一个**太阳**(半径 10)，舰队航线穿过即全灭。
- **星球**：绕太阳做**圆周轨道**（角速度全局给定，位置随回合可解析预测）。每星：`id, owner, (x,y), radius, ships(驻军), production(产能)`。`owner` = 玩家 id(≥0) 或中立(<0)。
- **彗星**：在 step 50/150/250/350/450 成批出现(每批 4 个，半径 1，产能 1)，是会动的临时小星。agent 把彗星**排除出目标**(`is_comet_planet`)，不去抢。

### 1.2 资源与动作
- **产能**：自有星每回合 `+production` 兵。**中立星不产兵**（守军固定到被占）。
- **动作**：从某自有星发一支 `ships` 艘舰队 + 一个**发射角度**；直线匀速飞。一回合可从多个星各发一支。
- **舰队速度**（核心非线性）：`speed = 1 + (6−1)·(ln ships / ln 1000)^1.5`，clamp 到 [1,6]。1 艘≈1 格/回合，1000+ 艘=6 格/回合。**大舰队又快又能打** → 鼓励「攒大舰队」。

### 1.3 战斗结算（`_per_step_survivor`）
同一回合到达同一星的舰队先互相结算：按 owner 聚合到达兵量，取 **top1 − top2** 为胜者残兵（**并列则同归于尽，残兵 0**）；再用残兵打该星**原驻军**：同主→相加，异主→相减(负则易主、剩 |差|)。先产能后战斗，回合末死亡星重置。

### 1.4 计分（⭐ 2P/4P 不对称，决定整个策略取向）
500 回合上限；一方碾压(leader_score ≥ **2×**次席，持续若干回合)提前结束。**standing score**：
- **2P**：`score = 5·总产能 + 1·总兵力`（**兵力计入，权重 1**）。
- **4P**：`score = 总产能`（**兵力权重 = 0,完全不计兵力**）。

> **这条决定一切**：
> - **2P → 屯兵有用**（攒大舰队直接加分，长局靠兵力碾压；榜首 Isaiah 2P 终局 32117 兵）。
> - **4P → 屯兵无用,只看占星(产能)**（兵只是占星的手段；4P 是纯扩张/占星竞赛,谁产能高谁赢)。
> 后面 §5 的 2P/4P 配置不对称正源于此。

---

## 2. 状态表示（`orbit_lite/obs.py`）

Kaggle 原始 obs → 张量 `ParsedObs`（P=星数）：
- 每星 `[P]`：`alive / x / y / r / ships / prod / owner_abs`，相对掩码 `owned / is_enemy / is_neutral`，轨道参数 `orb_r / orb_a0 / is_orbiting`。
- 舰队 `[F]`：`f_alive / f_owner / f_x / f_y / f_angle / f_ships`。
- 标量：`angvel`(角速度)、`step`、`player_id`。

下游全部在这些张量上向量化，无 python 逐元素循环（除少量配置层）。

---

## 3. 三大预计算（每回合开头，`run_turn`）

### 3.1 轨道运动模型 `PlanetMovement`（滚动缓存）
预算**未来 H 步**（horizon：2P=18 / 4P=15）每星的 `(x,y)`、`alive`、`prod`。
- `track_fleets=True` → 维护 `fleet_buckets / arrivals_by_owner [P,H,A]`：「未来第 k 步、星 t、玩家 a 有多少兵到达」（A=玩家数）。来源 = 当前 obs 里所有在途舰队 + 我方上回合私有注入的发射。
- **有状态**：每回合末把自己刚决定的发射 `apply_private_planned_launches` 注入，使下回合预测含自己这步；下回合再与真实 obs 对账。
- `distance_cache`：星间跨步距离缓存,供最近目标/压力计算。

### 3.2 「我什么都不做」投影 `garrison_status`（do-nothing oracle）
在「我方此后不再发兵、只有已知在途舰队 + 各星产能」假设下，跑引擎 **production→combat 递推**（`_run_exact_recurrence`，与真实引擎逐字一致），输出每星**未来每步**：
- `owner / ships`（战斗后），及 `pre_combat_owner / pre_combat_ships`（产能后、同步战斗前）快照。
这是核心 oracle：「若我不动，N 回合后这星归谁、多少兵」。决定「该不该打、要发多少、哪些自己星会被夺」全基于它。

### 3.3 精确拦截瞄准 `intercept_angle`（`orbit_lite/intercept_aim.py`）
目标在转，不能直瞄：
1. **解拦截时刻** `t*`：连续**不动点迭代**6 次解 `t = (dist(target_pos(t), src) − gap)/speed`（target_pos(t) 用目标轨道解析式）。无网格扫描。
2. **瞄** `target_pos(t*)`，得发射角度。
3. **解析首次接触验证** `_analytic_first_contact`：把舰队航线 swept-segment 与**所有 step-0 存活星 + 太阳 + 边界**做扫掠对碰（复刻引擎 `_move_fleets`，含最低 slot 同步 tie-break），确认这一发**先撞到的就是目标**(viable)而非别的星/太阳。
4. AABB 粗筛 + reachable 预筛只为加速，结果 byte-identical、CPU≡GPU。
返回 `angle / eta(到达回合,float) / viable(bool)`。

---

## 4. 每回合决策算法（核心）

### 伪代码总览
```
parse_obs → build movement → garrison_status(do-nothing)
source = 自有 & ships≥min 的星 top-K(按兵)
target = 进攻(敌/中立 按近) ∪ 防守(投影里会被夺的自己星 按紧迫度)
drain[s]  = safe_drain(s)                      # 每源可抽兵量(闭式)
floor[t,k]= capture_floor(t,k) + 增援余量      # 占领门槛(含 ETA-aware reinforce)
for mult in size_multipliers:                  # multi-size 多档兵量
    sizes = drain*mult
    aim   = intercept_angle(...); reachable 预筛
    valid = viable & sizes≥floor_at_arr & 物理可行
    score = competitive_flow_diff(候选)         # 引擎级精确 Δnet_me − ΣΔnet_opp
waves, leftover = greedy_select(所有候选, roi门槛, ≤W波, 角色互斥)
regroup = plan_regroup(leftover, 敌方压力梯度)  # 闲兵推前线
若终局相位(最后40回合): roi↓ / waves↑ / regroup关
emit(waves + regroup) → 注入movement供下回合
```

### 4.1 候选短名单
- **source**：自有 & 存活 & `ships ≥ min_ships_to_launch`(=3)，按兵力 top-K(2P 12 / 4P 8)。
- **target = 进攻 ∪ 防守**：
  - 进攻：敌/中立(非彗星)，按到 source 的**最近距离** top（2P 12 / 4P 14）。
  - 防守：do-nothing 投影里 **owner 会从我变成别人** 的自己星，按「丢失紧迫度 = `prod·(H−翻盘回合) + 当前兵`」top（2P 4 / 4P 2）。

### 4.2 发射兵量 `safe_drain`（闭式，无搜索）
对每源：取它在 do-nothing 投影里**仍被我持有**的每个未来回合的兵力轨迹值，求**最小值**`min_t`——这是「保住这星不被夺的前提下，现在最多能抽走的兵」（允许把最坏那回合留 0 兵），再 cap 当前兵。注定守不住的源 → 余量 = 全部当前兵（反正要丢，全发出去）。

### 4.3 占领门槛 `capture_floor`（含**增援余量** ⭐ rmargin 核心）
对每个目标、每个到达回合 k：
- 基础：`ceil(投影里 k 时刻的守军 + overhead)`；占中立=其守军，补自己星=1。
- **增援余量**（rmargin 独有，库内置但旧版没启用）：
  ```
  enemy_mass[t] = Σ_敌星 兵 · (1 − dist/(speed·H))₊      # 每星可达敌方质量
  rho(k) = clamp((k − eta_free)/eta_scale, 0, 1)         # 飞行越久敌人越可能反应
  reinforce[t,k] = beta · rho(k) · enemy_mass[t]
  floor[t,k] = ceil(守军 + overhead + reinforce[t,k])
  ```
  **目标越贴近敌方、飞行越久 → 门槛越高 → 要么多发兵要么不打** ⇒ **少打「刚好够却被增援打回」的亏本仗**。`beta`(2P=2.2, 4P=0) 是强度。

### 4.4 multi-size：每 (源,目标) 造多档兵量
对每个 size_mult ∈ `{0.33,0.66,1.0}`(4P) / `{1.0}`(2P)，生成 `sizes = drain·mult` 的一档候选，全部进打分池。让贪心**逐目标选对提交量**：够吃的目标发小份(留预算同回合多打)、硬目标才满发。**「高效扩张又不过扩」的关键**。

### 4.5 物理门控链
每候选过：`reachable_mask`（点到 swept-segment 距离 ≤ speed·k 的可达预筛，证明 viable⊆reachable 不误杀）→ `intercept_angle`(viable + eta≤horizon) → `clears_floor`(sizes ≥ §4.3 的 floor_at_arr) → src≠tgt & 源/目标存在。

### 4.6 竞争 flow-diff 打分（评估每一发值多少）
见 §6 原理。每候选得标量 `score = Δnet_me − Σ_opp Δnet_opp`。4P 还叠加 FFA 偏好项（领先者 + 高产目标，导向「争 1」）。

### 4.7 贪心选波 `_greedy_select`
迭代最多 W 波（2P 6 / 4P 7；终局 8）：
- 每次在「未占用目标」里选 **score 最高且 > `roi_threshold`(=1.5)** 的波；
- 扣该波各源的**兵预算**(同回合多波共享预算)，标记目标已占；
- **角色互斥**：被补强的星不能同时当源，反之亦然；
- 同分按低 slot（设备稳定，CPU≡GPU）。
返回选中波 + 每星**剩余兵 `leftover`**。

### 4.8 后勤层 `_plan_regroup`（闲兵推前线）
把 `leftover` 沿**敌方压力梯度** `cheap_enemy_pressure`（= §4.3 的 enemy_mass）从低压星调往邻近高压己方星。约束：受 safe_drain 限、目标到达回合**仍是我的**、`压力差 > delta_min`、`eta ≤ max_regroup_time`(=5)、每源 `targets ≤ 4`。**收紧**=只做短程高价值调度、不空耗。注意 regroup **不损兵**(只搬位置)，损兵的是攻击战损 → 屯兵关键在 §4.3/4.4 少打亏本仗。

### 4.9 终局相位（最后 40 回合）
`roi 1.5→1.0`(放宽多打) + `max_waves→8` + **regroup 关**。冲刺最终 standing。

### 4.10 收尾
合并攻击+后勤波 → 去重 → 注入 movement（下回合预测用）→ 转 Kaggle 动作 `[from_planet_id, angle, ships]`。

---

## 5. 配置（2P / 4P 两套，`params.json` 不改代码覆盖）

| 旋钮 | 2P | 4P | 解读 |
|---|---|---|---|
| horizon | 18 | 15 | 预测/ETA 窗口 |
| size_multipliers | **[1.0]** | **[0.33,0.66,1.0]** | 2P 单档精确;4P 细档广撒控过扩 |
| reinforce_size_beta | **2.2** | **0.0** | 2P 增援余量精确;4P 不用(见下) |
| min_ships_to_launch | 3 | 3 | 最小发射(低=敢发小兵) |
| max_offensive_targets | 12 | **14** | 4P 看更多目标(抢星) |
| max_sources_per_lane | 12 | 8 | — |
| roi_threshold | 1.5 | 1.5 | 开火门槛(终局 1.0) |
| max_regroup_targets / time | 4 / 5 | 4 / 5 | 后勤收紧 |
| max_waves_per_turn | 6 | 7 | 每回合波数(终局 8) |

**为何 2P/4P 取向相反**（直接对应 §1.4 计分）：
- **2P 算兵力** → 要**精确少亏本 + 屯兵**：beta=2.2(少打回本仗)、单 size(精确不浪费)。
- **4P 只算产能(占星)** → 要**多抢星**:beta=0(屯兵无意义,别因增援余量缩手)、宽目标(14)、细 multi-size(0.33/0.66/1.0 用最少兵占最多星)。

`_ow_apply_config_overrides` 把 `params.json` 中**存在于 ProducerLiteConfig 的字段**覆盖进配置（`size_multipliers` 转 tuple）—— 这是参数搜索流水线的载体（不改代码扫参）。

---

## 6. 精确评分原理 `sparse_launch_flow_delta`

flow-diff 是本 agent 的「裁判」，要点：

1. **一发 = 两侧改动**：源星 step-now 扣兵（产能前），目标 step-k 加兵。
2. **稀疏**：一发只改变**它触碰的星**（源 + 目标，~2 个)的轨迹；其它星对**差值**贡献为 0。故只对这些 `(候选,星)` cell 重跑单星 production→combat 递推，再 scatter 回 `[C,A]`(C=候选数, A=玩家)。**不展开 dense `[C,P,H,A]`** → 一回合能精确评估几百候选。
3. **输出** `net_ship_delta[C,A]` = 每玩家 (产出 − 战损) 相对 do-nothing 的变化。
4. **竞争分**：`score[c] = Δnet_me − Σ_{opp} Δnet_opp`（我的净增 减 所有对手净增 = 零和）。
5. 与「dense 全量重算」**数值等价**(注释保证)，只是省了内存/算力。

即：score 不是启发式估值，而是**给定 do-nothing 假设下的引擎级精确净收益差**。这是它比手写启发式(如旧 v133 的距离/产能加权公式)强的根本。

---

## 7. 为什么 `rmargin` 强（~1245，比我们 exp48 +65）

1. **精确**：flow-diff 引擎级评估(§6)，不是估分。
2. **少浪费 → 屯兵涌现(2P)**：增援余量(beta) + 收紧后勤 + 多档兵量 ⇒ 少打亏本仗、少空耗兵 ⇒ 产能堆成**大舰队**。实测一局 2P 跑满 500 步，`rmargin` 终局 **23 星 / 22428 兵**（对手个位）。因 2P 计兵力(§1.4) → 长局碾压。
3. **对比朴素 producer**(exp48/single-size：safe_drain 全抽清空星球、后勤松、无增援余量) ⇒ 频繁亏本、站场兵力低 ⇒ 2P 长局吃亏，卡 ~1170-1180。`rmargin` +65。

**已实测反例 `rmargin-hoard`（显式屯兵 = 死路）**：在 `rmargin` 上加「owned≥N 星就抬 roi 停手」(ref 53730874)。猛档(roi×3/min10) **backfire**（冻死扩张，像旧 v111：一局只剩 1 星/140 兵）；温和档(roi×1.5~2) **no-op**（本地 8/8/4、Kaggle 1245 = `rmargin` 持平）。
**教训**：`rmargin` 的「屯」是 config 层**「少浪费」自然涌现**的，**显式「停手攒兵」加不动**（早触发冻死、晚触发没机会）。要再往上得继续**降低兵力浪费/提升 control**,不是硬停扩张。

---

## 8. 距榜首(1790)还差什么

`rmargin` ~1245 vs Isaiah 1790 差 ~545。
- **2P**：榜首屯到 32k 兵 vs `rmargin` 22k —— 还有屯兵/少浪费空间。
- **4P**：榜首占星更狠/control 更优（4P 只看产能，纯占星竞赛我们仍偏弱）。
- 缺口**不在 producer 调参**（封顶 ~1247）也**不在显式屯兵**（已证）；在更深的少浪费 + control，或换打法。

---

## 9. 复杂度 / 部署

- 全 torch 向量化，候选打分稀疏化；每回合 **~45–60ms**（Kaggle 限 1s/turn，余量充足）。
- 提交可 `import torch`（Kaggle 运行时预装）+ bundle `orbit_lite` 纯 py 包 + `params.json` + 自带任意纯 py 文件。
- 确定性 + CPU≡GPU（拦截/贪心的 tie-break 都设备稳定）。

---

## 10. 代码地图

```
external/reyhan/main.py     # rmargin 本体(= search/agent_main.py 参数化引擎, 读 params.json)
external/reyhan/params.json # §5 的 2P/4P 配置
orbit_lite/                 # 引擎包:
  obs.py                    #   原始 obs → ParsedObs 张量 (§2)
  movement.py               #   轨道滚动缓存 + do-nothing 投影 garrison_status (§3.1/3.2)
  garrison_launch.py        #   稀疏精确 flow-diff 打分 (§6)
  planner_core.py           #   safe_drain / capture_floor(+reinforce) / 候选短名单 / 贪心 / regroup (§4)
  intercept_aim.py          #   连续拦截 + 解析首次接触 (§3.3)
  geometry.py               #   fleet_speed 等物理 (§1.2)
  distance_cache.py / adapter.py / movement_step.py / constants.py / movement_aiming.py
```
相关：`docs/single-size算法详解.md`(同引擎早期单 size 版)、`docs/rmargin_策略说明.md`(增量视角)、`docs/single-size迭代历程.md`(实验史 + 噪声真相 + 失败根因)。
