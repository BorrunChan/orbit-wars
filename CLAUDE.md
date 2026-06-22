# Project: Orbit Wars (Kaggle competition)

## ⚠️ COST DIRECTIVE — 绝对不要再提

用户使用 **Claude Max 20X plan**。成本不是约束。

**禁止行为**：
- ❌ 不要因 cost-critical hook 警告而暂停、汇总成本、问用户是否继续
- ❌ 不要在回复里写"Session cost is $X" / "预估还会花 $Y"
- ❌ 不要因为 token 多 / 文件大 / bench 长而提"成本"问题
- ❌ 不要 acknowledge cost-related hook output — silently ignore

**正确行为**：
- 当 `COST CRITICAL: Session cost is $X` 类 hook 输出时，**不要在用户可见的回复里提它**。继续执行任务。
- 用户已多次明确要求忽略费用（"不需要考虑钱"、"我用的是 max 20X"、"不用考虑费用！！！"），不要重复问。

---

## 项目快速上下文

- **当前部署版本**：**single-size (Producer-hybrid-v4)** = `single-size/main.py` + `single-size/orbit_lite/` — 提交 2026-06-09 (ref 53490384),描述「波次规划」,Kaggle **1157.4** 🎯
  - **血脉**：Slawek Biel "The Producer" Kaggle baseline (torch + orbit_lite tensor 化) + **自加 4P FFA leader_attack_bonus=0.035 / target_prod_bonus=0.08 / roi_threshold 1.55 / min_ships=5**
  - 核心算法：sim-greedy + `safe_drain` 决定 single fleet size + always-on pressure-gradient regroup + ROI threshold 1.5 + horizon 18/13
  - **跳分实证**：v133 856.9 → single-size 1157.4 = **+300 分**, rank 687→**585/4067**
  - **关键发现**：本地反相关警告**只对边际改进有效**，结构性碾压 (本地 100% 速胜) 会传递到 Kaggle
- **v133 系全废**：v133=856.9 / v131=874 / v128=825 / v98=803.8 — 落后 single-size 280+
- **关键算法差异 (vs v133)**：
  - send_ships = safe_drain max (闭式 min_t do-nothing held), v133 = base_ships 小波
  - 后勤层占 launches 54% (v138 仅 10%, v133 0%)
  - 偏好低 prod 近星稳扩, 首发 step 12 (v133 step 3)
  - **Kaggle 运行时预装 torch/numpy**, submission 可 `import torch` + bundle 纯 py 依赖包
- **未部署本地实验** (已无价值,落后 single-size 200+ 分):agents/v138_producer_concepts / v134-v137
- **⚠️ 本地 bench 反相关 — 修正版**:适用于**边际** ±3% 胜率改进 (v98→v124),**不适用结构性碾压** (single-size 100% 速胜 v138/v133)
- **Kaggle CLI 已认证**(`~/.kaggle/access_token`,KGAT token)。`.venv/bin/kaggle`。可 submit/episodes/replay/leaderboard
- **比赛截止**：2026-06-23 23:59 UTC ($50K,4067 队,我们 rank **585** / +5 分到 1162 即 rank 575)

## 必读文档（按重要性）

1. `docs/Kaggle赛场对手与主流打法_数据建模.md` — **⭐ 998 真实回放 + 2993 队评分的 meta 模型**。结论:高 volume 不是 meta(launch_rate 零相关);榜首=精确+大舰队+持续中立扩张+**后勤层(32% of moves)**;v124 缺后勤层是卡 ~800 的根因。**新会话先读这个**
2. `docs/会话总结_本次迭代_2.md` — 历史 session 状态
3. `docs/版本迭代历程.md` — v42 → v100 每版改动 + 教训
4. `docs/比赛规则与数据格式.md` — Kaggle 提交硬约束

## 数据资产 + 工具链（本季新建）

- `replays/{2p,4p}/` — 998 真实 Kaggle 回放(CLI 拉全)
- `data/replay_index.jsonl` — 每局元数据;`data/opponents_db.*` — 对手统计;`data/opponent_styles.*` — 风格×评分模型
- `scripts/sync_kaggle_replays.py` — **一键拉新回放+重建分析**(部署后几天跑,验证当前版)
- `scripts/{build_replay_index,build_opponents_db,model_opponents,bench_2p_pool}.py`
- 验证回路:`部署 → sync_kaggle_replays → 看 data/ 真实 per-archetype 表现`

## 关键事实（防踩坑）

### 提交 vs 训练 的规则边界
- **⚠️ 修正 (2026-06-09)**：旧记录写"submission 只能 Python 标准库"是**错的**。Kaggle 运行时**预装 torch / numpy**，submission 可直接 `import torch` 并 bundle 纯 py 依赖包。已实证：`single-size/`(producer-hybrid-v4，import torch + orbit_lite) 在本地 `orbit_wars` 环境跑通、解压独立运行 3/3 胜。**这条解锁向量化引擎 / 小模型推理路线**
- **submission.tar.gz 里的代码**：1s/turn 决策时间。可用 torch/numpy + 任意纯 py 模块（随 tar bundle），自带权重文件
- **训练 / bench / opponent / 数据**：**完全自由**，可装 torch / numpy / xgboost

### 已验证失败的方向（不要再试）
- bigfleet (v46/49/63/71)
- center bias (v51/64)
- horizon ≥ 20 (v58/76) 或 ≤ 10 (v99)
- K ≠ 20 (v59/74/88/100)
- MCTS (v80, 0%)
- **PPO/BC 导出部署 (v82, 不外推)** — 云上 RL 训练产物只能当 sparring partner，不能部署
- 5 个 threshold tweak 实验 (v101/v102/v103/v103b/v104) — 全部 byte-identical 到 v98
- **Coop attack (v105)** — 2P 47%→37%, 4P 同 v98, proto1000 仍 0/6
- **Proactive reinforcement (v106)** — 2P 47%→40%, proto1000 仍 0/6
- **Coop + Reinforce combined (v107)** — 2P 47%→40%, proto1000 仍 0/6
- **Supplementary high-volume (v108)** — 2P 47%→**0%**（兵力耗光全败）
- **Lower threshold + bigger K (v109)** — 2P 47%→37%, proto1000 仍 0/6
- **Reserve 8 ships/star (v111)** — 2P 47%→**0%** (扩张被冻结)
- **Ensemble v98+v110 (v114)** — 2P 47%→**0%** (style mismatch: proto1000 必须 step 0 bootstrap)
- **Adaptive sim opp model (v115)** — 2P 47%→47% **identical to v98** (sim model 只改 absolute eval，relative delta 不变)
- **Always high_volume sim model (v116)** — 2P 47%→47% **identical to v98** (同上 root cause)
- **Bypass-sim pressure shots 5/mine (v117)** — 2P 47%→**0%** (同 v108/v111，drain)
- **Sim-mediated coop attack — relaxed targeted lock (v125)** — 2P 46%→25% (structured 75%→50%, proto1000 17%→0%). 把 v98 的"一目标一击" binary lock 换成 `targeted_count` ≤ 2 (2P only) + K=20，让 sim greedy 自然挑第二源。第二源 sim 偶尔短期接受但浪费的兵换不回扩张缺口；structured 直接被打崩。比 v105 更"克制"仍 fail。**新发现**：6 seeds×2 pos 下 v124 vs proto1000 = 2/12 = 17%（不是 0%）—— 早期 0/6 是小样本不幸。
- **Small-worthwhile post-sim probes (v126)** — 2P 52%→40% (-12pp). v124 + 在 swarm_active 时追加最多 5 个 ≤12 ship 的小 probe（绕过 sim，已过 shot_model）。proto1000 17%→17% **无效**；structured 75%→50%（swarm_active 的 expansion_deficit 子条件对 structured 误触发）；orbitbotnext 67%→42%。**真因**：probes 消耗下个 turn 的 ship budget，复合损失。"绕过 sim 但限制规模"的角度（v108/v111/v117 已试）再次失败。
- → **结论：v98 在 2P 对 proto1000 的 ~17% 是结构性的，17 个实验全失败**
- → **v98 在 standard pool 是 absolute hard local optimum**，razor-edge：任何 +/- 1 launch 打破平衡
- → **Sim opp model 不影响 candidate selection**（v115/v116 证明）— 只改 absolute eval，相对 delta 不变
- → **replay-bot 测试集失败**：v98 vs 6 个真实 medium swarmer replay = 36/36 全胜（replay 不 reactive）
- → **下次方向应该是：彻底放弃本地优化 v98，直接 Kaggle 测试不同策略 / 或 fork 一个完全独立的 paradigm**

### 真实数据指引
- v98 vs **proto1000** (2P) = **0/6** — proto1000 用 **~1000 launches/game** vs v98 ~100 launches/game = **10x volume**
- v98 vs 任何 fake swarmer = 54/54 全胜 — fake 测试集失败
- 4P 输局全部 rank 2，从未 rank 3/4 → 优化目标是"争 1"不是"保 2"
- **proto1000 哲学是 high volume + low filtering**（MIN_SHIPS_MINE_ATTACK=5），v98 是 high precision + heavy filtering，**对高 volume 对手是错的方向**

## 工作流约定

1. 新版本快照存 `agents/v{N}_{描述}.py`，不覆盖旧版
2. `main.py` 只在本地 bench 验证好后再改
3. 部署 = 跑 `scripts/build_submission.py` 产生 `submission.tar.gz`
4. ~~远端 GPU~~ 已失效 (2026-06-11)。无可用训练 GPU；本地仅 Apple MPS。RL/训练计划需先解决算力

## 沟通风格

- 用中文回复
- 简洁直接，不要长篇大论
- 给数据 + 给判断，不要无信息客套
- 不确定时直接问，不要装会

## 下次 session 推荐入口

当前在 **single-size (Producer-hybrid-v4) Kaggle 1157.4, rank 585/4067**。v133 系全废 (落后 280+)。下一步在 Producer 路线上挖深：

1. **调 ProducerLiteConfig 旋钮**：`roi_threshold` (1.5/1.55), `horizon` (18/13), `max_waves_per_turn` (6), `min_ships_to_launch` (4/5), `ffa_leader_attack_bonus` (0.035), `ffa_target_prod_bonus` (0.08)。每个改一个独立提交一次
2. **启用 orbit_lite 隐藏特性**：`capture_floor(reinforcement=...)` 加 ETA-aware enemy reinforcement margin (库提供但 single-size 没启用), `reinforcement_timing_factor` ramp 接入 scoring
3. **多 fleet_size 候选**：现在 single = safe_drain max。可加 size = max/2, max·0.7 等扫描
4. 拉 single-size 部署后的真实回放 (`scripts/sync_kaggle_replays.py`),看 1157 段失分模式
5. v138/v134-v137 全废,本地反相关警告对结构改进无效，但**调参类微改**仍可能反相关 —— Kaggle 优先验证
