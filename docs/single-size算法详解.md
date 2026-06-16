# single-size (Producer-Hybrid) 算法详解

> 当前主线算法。部署版 `single-size/main.py` + `single-size/orbit_lite/`,Kaggle **~1180**(2026-06-09,ref 53490384),较旧主线 v133(~855)**+325**。
> 血脉:Slawek Biel 公开的 "The Producer" baseline(torch + orbit_lite 张量化引擎)+ 自加 4P FFA 调参。

---

## 0. 一句话概述

一个 **全 torch 向量化的 sim-greedy 波次规划器**:用引擎级精确前向模拟(flow-diff)给每个候选发射打"我净收益 − 对手净收益"的竞争分,贪心选出每回合最优的几波攻击,再把剩余兵力沿敌方压力梯度做后勤调度。

与旧主线 v133(标量 Python 启发式)同源(都是 sim-greedy + 射击验证),但做了三件 v133 没有的事:**全向量化**、**解析稀疏精确打分**、**带后勤层**。

---

## 1. 三大基石

### 1.1 轨道运动模型 `PlanetMovement`(`orbit_lite/movement.py`)
- rolling cache:预算所有星球未来 `H` 步的位置 `x/y`、存活、产能。
- `track_fleets=True`:追踪所有在途舰队,得到 `arrivals_by_owner [P, H, A]` —— 未来每一步、每个目标、每个 owner 各有多少兵到达。
- stateful:每回合把自己已规划的发射经 `apply_private_planned_launches` 私有地注入模型,使下回合投影包含自己的动作。**player_count 在 step 0 缓存**(决定用 2P 还是 4P config)。

### 1.2 Do-nothing 投影 `garrison_status`
- 在"我什么都不做"假设下,跑引擎的 **production→combat 递推**(`_run_exact_recurrence`,逐字复刻引擎),得到每星未来每步的 owner/ships(post-combat)+ pre-combat 快照。
- 这是 oracle:"N 回合后这里是谁的、多少兵"。是 `safe_drain` / `capture_floor` / 防守目标判定的基础。

### 1.3 精确拦截瞄准 `intercept_angle`(`orbit_lite/intercept_aim.py`)
- 目标在轨道上转 → 不能直瞄。**连续不动点迭代**(6 次)解拦截时刻 `t*`,瞄 `target_pos(t*)`。
- **解析首次接触检查** `_analytic_first_contact`:swept-pair vs 所有星/太阳/边界 + 最低 slot 同步 tie-break,逐字复刻引擎 `_move_fleets`,验证这一发会不会先撞别的星/掉太阳。
- AABB 粗筛 + `reachable_mask` 预筛保证 byte-identical 不误杀。返回 `angle / eta / viable`。

---

## 2. 每回合决策流水线(`plan_lite_waves` → `run_turn`)

```
build movement → garrison_status(do-nothing 投影)
  → 候选生成(source × target shortlist)
  → 门控链(reachable → capture_floor → intercept viable)
  → 竞争 flow-diff 打分(+ 4P FFA bonus)
  → 贪心选波(roi 门槛, ≤ W 波)
  → 后勤层 regroup(剩余兵沿压力梯度)
  → emit moves
```

### 2.1 候选生成
- **source**:自有存活、`ships ≥ min_ships_to_launch` 的星,按兵力 top-K(2P 12 / 4P 6)。
- **target** = 进攻 ∪ 防守(`build_target_shortlist`):
  - 进攻:敌/中立按距离近,top `max_offensive_targets`(2P 12 / 4P 7)。
  - 防守:do-nothing 投影里会翻盘的我方星,按"丢失紧迫度 = prod·剩余回合 + 当前兵",top `max_defensive_targets`。
- **每个 (src,tgt) 只造一个候选**,fleet size = `safe_drain`(闭式解:保住源星不被 do-nothing 投影翻盘的前提下最多抽多少兵 = 未来每个仍持有回合的兵力轨迹最小值)。

### 2.2 门控链
1. `reachable_mask`:点到 swept-segment 距离 ≤ speed·k 的严格超集可达性预筛(证明 viable ⊆ reachable,不误杀)。
2. `capture_floor`:守军随到达回合 `k` 增长(中立星是静态守军,敌星含产能增长);发射量必须 ≥ 到达那刻守军 + overhead。我方目标=补强,floor=1。
3. `intercept_angle` viable + `eta ≤ K_eta`。

### 2.3 竞争 flow-diff 打分(`sparse_launch_flow_delta` + `competitive_score`)
- 一次发射 = 源星扣兵 + 目标第 `k` 步加兵。
- **只对被触碰的 2 个星(源+目标)重跑引擎递推**,和 baseline(do-nothing)做差 → 每玩家 Δ(产出 − 战损)= `net_ship_delta`。
- 稀疏:不展开 dense `[C,P,H,A]`,只算 affected cells scatter 回 `[C,A]` → 一回合能精确评估几百个候选。
- **分数 = `Δnet_me − Σ_opp Δnet_opp`**(零和竞争分)。

### 2.4 贪心选波 `_greedy_select`
- 迭代 ≤ `max_waves_per_turn`(=6),每次在未占用 target 里挑分最高且 `> roi_threshold` 的波,扣源星预算,标记占用。
- **角色互斥**:被补强的星不能同时当源,反之亦然。设备稳定(CPU≡CUDA tie-break)。
- 返回:选中波 + 每星剩余兵 `leftover`。

### 2.5 后勤层 `_plan_regroup`(v133 缺失的一层)
- 把 `leftover` 沿 **敌方压力梯度**(`cheap_enemy_pressure`:距离衰减的可达敌方质量)从低压星调往邻近高压己方星。
- 约束:受 `safe_drain` 限制、目标到达回合仍是我的、压力差 > `regroup_pressure_delta_min`、`eta ≤ max_regroup_time`。
- 对应 meta 模型里"榜首占 ~32% moves 的 logistics"。

---

## 3. 2P / 4P 分治(`ProducerLiteConfig` / `CONFIG_4P`)

| 旋钮 | 2P 默认 | 4P (`CONFIG_4P`) | 含义 |
|---|---|---|---|
| `horizon` | 18 | 13 | 投影窗口 = 运动构建长度 = ETA cap |
| `max_sources_per_lane` | 12 | 6 | source shortlist 宽 |
| `max_offensive_targets` | 12 | 7 | 进攻目标宽 |
| `max_defensive_targets` | 4 | 2 | 防守目标宽 |
| `roi_threshold` | 1.5 | 1.55 | 开火分数门槛 |
| `min_ships_to_launch` | 4 | 5 | 最小发射兵力 |
| `max_waves_per_turn` | 6 | 6 | 每回合最多波数 |
| `max_regroup_time` | 7 | 6 | 后勤最大跳长 |
| `ffa_leader_attack_bonus` | 0 | 0.035 | 4P:对领先者加分 |
| `ffa_target_prod_bonus` | 0 | 0.08 | 4P:对高产目标加分 |

**4P FFA bonus**:`ffa_leader_attack_bonus·(领先者强度 − 我) + ffa_target_prod_bonus·目标产能`,把火力导向第一名 + 高产星(**争 1,不保 2**)。4P 整体更保守(短 horizon、高 roi、窄 shortlist),避免掏空星被第三方夺(R4 问题)。

---

## 4. 已知缺陷 + 实验分支(2026-06-09)

### 4.1 核心缺陷:4P 系统性"不攻击"
- 62 局真实回放:WR 44%(**2P 46% / 4P 仅 42%**)。
- 量化证据:被动 4P 负局里 **57–63 个闲置回合手握可达可清目标却拒绝攻击**(vs 仅 3–24 真无目标)。**是评分缺陷,非几何**。
- **根因**:competitive flow-diff 在 4P(horizon=13 + roi 1.55)只信用窗口内 ~5 回合产能,把可清的中立扩张算成净亏船;中立捕获在 4P 不削弱对手 → 无 competitive credit → 坐 3–6 星等死。

### 4.2 实验分支(同目录变体)
| 变体 | 改动 | 结果 |
|---|---|---|
| `main.py`(基线) | Producer-hybrid-v4 | **Kaggle 1180,当前最优** |
| `main_bootstrap.py` | owned≤1 时去 roi 强制扩张 | 单变体,未单独提交 |
| `main_coop.py` | 多源 coop(L>1 contributor) | 单变体 |
| `main_v2.py` | bootstrap + coop(全开) | Kaggle 1095,**净负**(coop 4P 掏空被夺) |
| `main_v3.py` | bootstrap + coop 限 2P | Kaggle 1100,**净负** |
| `main_expand.py` | 4P 中立扩张价值 `neutral_expand_bonus·prod`(=3.0),仅 4P | 提交中(ref 53501124) |

**教训**:
- **bootstrap / coop 在 4P 增扩 → 过扩被团灭 → Kaggle 负**(撞 v133 R4 老问题)。
- 本地 bench 饱和(~98%)、反相关,**这些结构-边缘改动只能 Kaggle 验证**。
- 待验证的开放问题:4P 低攻击究竟是缺陷,还是**正确的保守**(苟 rank 2–3 优于扩张暴毙)?`main_expand` 的分数将给出答案。

---

## 5. 文件地图

```
single-size/
├── main.py                 # 部署基线(Producer-hybrid-v4),Kaggle 1180
├── main_expand.py          # 4P 中立扩张价值(当前实验,ref 53501124)
├── main_{bootstrap,coop,v2,v3}.py   # 历史实验变体
├── build_submission.py     # 打包(--main <file> --out <tar>)
└── orbit_lite/             # 向量化引擎包(Kaggle 数据集 slawekbiel/producer-orbit-wars-utils, CC0)
    ├── movement.py         # PlanetMovement + garrison_status(投影)
    ├── garrison_launch.py  # sparse_launch_flow_delta(稀疏精确 flow-diff)
    ├── planner_core.py     # 候选/打分/贪心/regroup/safe_drain/capture_floor
    ├── intercept_aim.py    # 连续拦截 + 解析首次接触
    ├── geometry.py         # fleet_speed 等
    ├── distance_cache.py / obs.py / adapter.py / movement_step.py / ...
```

### 提交约束(已验证)
- Kaggle 运行时**预装 torch / numpy**;submission 可 `import torch` + bundle 纯 py 依赖包 + 自带权重。1s/turn 决策时间。
- 打包:`python single-size/build_submission.py --main main_expand.py --out submission_expand.tar.gz`
- 提交:`.venv/bin/kaggle competitions submit -c orbit-wars -f <tar> -m "<msg>"`
