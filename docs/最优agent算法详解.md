# Orbit Wars 当前最优 agent `rmargin` —— 完整算法详解（从零）

> **命名 `rmargin`**（reinforcement margin，增援余量版）。对应我们当前 Kaggle **public ~1245**（1241–1247，最高提交 ref 53704199，代码 `external/reyhan/` + 参数化副本 `search/agent_main.py`，引擎包 `orbit_lite/`）。
> 衍生 `rmargin-hoard`（+ 显式屯兵旋钮，ref 53730874）实测 = 1245 持平、无增益（§6）。
> 本文**自包含、从零讲**：游戏规则 → 状态表示 → 每回合如何决策 → 评分 → 配置 → 为何强。不假设读者了解任何先前版本。

---

## 0. 一句话

每回合，对"我所有星 × 附近所有可打目标"的组合，各假设"发一支舰队过去"，用一个**引擎级精确前向模拟**算出"这支舰队让(我方净收益 − 对手净收益)变化多少"，贪心挑出最划算的几支发出去；剩余兵力沿"敌方压力梯度"向前线调度。全程 torch 张量并行。

---

## 1. 游戏与物理（规则基础）

- **棋盘**：100×100 连续平面，正中 (50,50) 有个**太阳**(半径 10)，撞上即舰队全灭。
- **星球**：每个星球绕太阳做**圆周轨道运动**（位置随回合变化，可解析预测）。属性：`id, owner, (x,y), radius, ships(驻军), production(产能)`。owner = 玩家 id（≥0）或中立(<0)。
- **产能**：自己拥有的星球每回合 +production 兵。中立星**不产兵**（守军固定，直到被占）。
- **发射舰队**：从自有星发一支 `ships` 艘的舰队，给一个**发射角度**；舰队直线匀速飞行。
  - **速度**（关键）：`speed = 1 + 5·(ln(ships)/ln 1000)^1.5`，范围 [1,6]。**兵越多飞越快**（1 艘≈1/格，1000+ 艘=6/格）。所以"大舰队又快又强"。
- **战斗**（到达同一星的同一回合结算）：把该回合到达该星的各 owner 兵量取 **top1 − top2** 为胜者残兵（并列则同归于尽），再用残兵打该星原驻军（同主则相加，异主则相减、负则易主）。
- **终局/计分**：500 回合上限；一方碾压（leader_score ≥ 2×次席，持续若干回合）会提前结束。score ≈ `5·总产能 + 1·总兵力`(2P)。**注意：兵力进了计分** → 长局里**站场兵力多者占优**（后面会看到这是胜负关键）。

---

## 2. 状态表示（`orbit_lite/obs.py`）

把 Kaggle 原始 obs 解析成张量 `ParsedObs`：每星的 `alive / x / y / r / ships / prod / owner_abs`，及相对己方的掩码 `owned / is_enemy / is_neutral`；舰队 `f_owner/f_x/f_y/f_angle/f_ships`。一切下游计算都在这些 `[P]`（P=星数）张量上向量化。

---

## 3. 三大预计算（每回合开头）

### 3.1 轨道运动模型 `PlanetMovement`（滚动缓存）
预先把**未来 H 步**（horizon，2P=18 / 4P=15）每个星球的 (x,y)、是否存活、产能都算好。并开启 `track_fleets`：追踪所有在途舰队，得到 `arrivals_by_owner [P, H, A]`——"未来第 k 步、星 t、玩家 a 有多少兵到达"。这是有状态的：把自己已决定的发射**私有地注入**模型，使下一回合的预测把自己这步也算进去。

### 3.2 "我什么都不做"投影 `garrison_status`
在"我方此后不再发兵"的假设下，跑引擎的 **production→combat 递推**（`_run_exact_recurrence`，与真实引擎逐字一致），得到每星**未来每步**的 `owner/ships`（战斗后）＋ 战斗前快照。这是 oracle："N 回合后这星会是谁的、多少兵"。后续的"该不该打、要发多少兵、谁会翻盘"全基于它。

### 3.3 精确拦截瞄准 `intercept_angle`（`orbit_lite/intercept_aim.py`）
目标在轨道上转，不能直瞄。用**连续不动点迭代**（6 次）解拦截时刻 `t*`（`t = (dist(target(t),src) − gap)/speed` 的不动点），瞄 `target(t*)`；再用**解析"首次接触"检查**（扫掠线段 vs 所有星/太阳/边界，复刻引擎 `_move_fleets` 的判定）验证这一发会不会先撞别的星或掉太阳。返回 `angle / eta(到达回合) / viable`。带 AABB 粗筛保证不误杀、CPU≡GPU 一致。

---

## 4. 每回合决策算法（核心，逐步）

### 4.1 候选生成：source × target 短名单
- **source（发射源）**：自有、存活、`ships ≥ min_ships_to_launch`(=3) 的星，按兵力取 top-K(2P 12 / 4P 8)。
- **target（目标）** = 进攻 ∪ 防守：
  - 进攻：敌/中立星，按**距离近**取 top（2P 12 / 4P 14）。
  - 防守：在 do-nothing 投影里**会翻盘（被夺）的我方星**，按"丢失紧迫度 = prod·剩余回合 + 当前兵"取 top（2P 4 / 4P 2）。

### 4.2 发射兵量 `safe_drain`（闭式，不靠搜索）
对每个源：在"保住该源不被 do-nothing 投影翻盘"的前提下，这一步**最多能抽走多少兵** = 未来每个"仍持有"回合的兵力轨迹**最小值**（可抽到那个最小值，把最坏回合留 0 兵也允许），再 cap 当前兵。一个注定守不住的源，余量塌到"全部当前兵"。

### 4.3 capture_floor：占领门槛（含**增援余量** ⭐）
对每个目标，算"到达回合 k 时它有多少守军 + overhead"作为**最低发射量**（占中立星=守军，补自己星=1）。
**关键增量**：再加一项**敌方增援余量**——
```
enemy_mass = 每星可达敌方质量（距离衰减）
rho(k) = reinforcement_timing_factor：飞行越久敌人越可能反应，∈[0,1] 的 ramp
reinforcement = beta · rho(k) · enemy_mass[target]      # [T, K]
floor = ceil(守军 + overhead + reinforcement)
```
即**目标越靠敌方、飞行越久 → 要求发越多兵才打**。作用：**少打"刚好够却被增援打回"的亏本仗**。`beta`(2P=2.2) 是强度旋钮。

### 4.4 multi-size：每个 (源,目标) 造多档兵量
不是只发"safe_drain 满量"，而是发 `safe_drain × {0.33, 0.66, 1.0}`（4P）或 `×{1.0}`（2P）三/一档候选，全部参与打分，让贪心**逐目标选对提交量**（够吃就发少、留预算；必要才满发）。这就是"高效扩张又不过扩"的关键。

### 4.5 门控链（物理可行性）
每个候选过三关：`reachable_mask`（点到扫掠段距离 ≤ speed·k 的可达预筛）→ `intercept_angle` 求角度+ETA+viable → `clears_floor`（发射量 ≥ 4.3 的 floor）。

### 4.6 竞争 flow-diff 打分（评估"这一发值多少"）
对每个候选，用 `sparse_launch_flow_delta`：一支发射 = 源星扣兵 + 目标第 k 步加兵；**只对被触碰的 2 个星重跑引擎递推**，与 do-nothing baseline 做差 → 每玩家 Δ(产出 − 战损) = `net_ship_delta`。
分数 = **`Δnet_me − Σ_opp Δnet_opp`**（零和竞争分：我的净增 减 对手净增）。
稀疏化（只算被影响的星、不展开 dense `[C,P,H,A]`）使一回合能精确评估几百个候选。

### 4.7 贪心选波 `_greedy_select`
迭代最多 W 波（2P 6 / 4P 7；终局 8）：每次在"未被占用的目标"里挑分最高且 `> roi_threshold`(=1.5) 的波，扣源星兵预算、标记占用。**角色互斥**：被补强的星不能同时当源、反之亦然。设备稳定（同分按低序号，CPU≡GPU）。返回选中波 + 每星**剩余兵 leftover**。

### 4.8 后勤层 `_plan_regroup`（把闲兵推向前线）
把 `leftover` 沿**敌方压力梯度**（`cheap_enemy_pressure`：距离衰减的可达敌方质量）从低压星调往邻近高压己方星。约束：受 safe_drain 限制、目标到达回合仍是我的、压力差 > 阈值、`eta ≤ max_regroup_time`(收紧到 5)、`targets_per_source ≤ 4`（**收紧**=只做短程高价值调度，不浪费）。

### 4.9 终局相位（最后 40 回合）
`roi 1.5→1.0`（放宽多打）、`max_waves→8`、**regroup 关**。冲刺最终名次。

### 4.10 收尾
合并攻击波 + 后勤波 → 去重 → 把自己这步注入 movement（供下回合预测）→ 转成 Kaggle 动作 `[from_planet_id, angle, ships]`。

---

## 5. 配置（2P / 4P 两套，`params.json` 可不改代码覆盖）

| 旋钮 | 2P | 4P | 含义 |
|---|---|---|---|
| horizon | 18 | 15 | 预测/ETA 窗口 |
| size_multipliers | [1.0] | [0.33,0.66,1.0] | 兵量档（2P 单档精确，4P 细档控过扩） |
| reinforce_size_beta | 2.2 | 0.0 | 增援余量强度 |
| min_ships_to_launch | 3 | 3 | 最小发射 |
| max_offensive_targets | 12 | 14 | 看多少进攻目标 |
| max_sources_per_lane | 12 | 8 | 看多少源 |
| roi_threshold | 1.5 | 1.5 | 开火门槛(终局 1.0) |
| max_regroup_targets / time | 4 / 5 | 4 / 5 | 后勤收紧 |
| max_waves_per_turn | 6 | 7 | 每回合波数(终局 8) |

`_ow_apply_config_overrides` 把 `params.json` 里存在的字段覆盖进配置——这是参数搜索的载体。

---

## 6. 为什么它强（1247）

1. **精确**：flow-diff 是引擎级精确评估（给定 do-nothing），不是启发式估分。
2. **少浪费**：增援余量(beta) + 收紧后勤 + 多档兵量 ⇒ **少打亏本仗、少空耗兵** ⇒ 产能堆积成**大舰队**。实测一局 2P 跑满 500 步，它终局 **23 星 / 22428 兵**——**屯兵是涌现的**（不是显式规则）。
3. **长局靠兵力计分赢**：因为 score 含兵力，攒下的大舰队在长局碾压。
4. 这比"清空星球猛攻"(safe_drain 全抽) 的朴素 producer 高 ~65（后者站场兵力低、长局吃亏，卡 ~1180）。

**注意（已实测的反例 `rmargin-hoard`）**：在 `rmargin` 之上加"owned≥N 星就抬 roi 停手"的**显式屯兵规则**（ref 53730874）。猛档 backfire（冻死扩张，像旧 v111：一局只剩 1 星）、温和档 no-op（Kaggle 1245 = `rmargin` 持平）。正确的"屯"是 config 层**降低浪费**自然涌现，而非"停手攒兵"。

---

## 7. 复杂度 / 部署

- 全 torch 向量化；每回合 ~45–60ms（远低于 Kaggle 1s/turn）。
- 提交可 `import torch`（Kaggle 运行时预装）+ bundle `orbit_lite` 纯 py 包 + `params.json`。
- 距榜首 1790 仍差 ~550，缺口主要在**更彻底的屯兵 / 更优 control**（榜首 32k 兵 vs 我们 22k vs 朴素 producer 几百）。

---

## 8. 代码地图

```
external/reyhan/main.py        # rmargin 本体（= search/agent_main.py 参数化引擎，读 params.json）
external/reyhan/params.json    # 上表的 2P/4P 配置
orbit_lite/                    # 引擎包：
  movement.py                  #   轨道缓存 + do-nothing 投影 garrison_status
  garrison_launch.py           #   稀疏精确 flow-diff 打分
  planner_core.py              #   safe_drain / capture_floor(+reinforce) / 贪心 / regroup
  intercept_aim.py             #   连续拦截 + 解析首次接触
  geometry.py / obs.py / distance_cache.py / adapter.py / movement_step.py
```
相关：`docs/single-size算法详解.md`（同引擎早期版）、`docs/reyhan_1241_策略说明.md`（增量视角）、`docs/single-size迭代历程.md`（实验史）。
