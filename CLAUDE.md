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

- **当前部署版本**：v124 (`main.py` = `agents/v124_early_swarm.py`)
  - v124 = v123 + 2P early-game max_launch floor (step 10-60, _n_total>=3, 25% floor)
  - v123 vs v124 standard pool 2P bench: 21/30=70% 同分 (trigger 不触发本地)
  - Kaggle 期望: 在真 swarmer 局 (fell_behind@10-30) 早期解锁兵力，覆盖 v123 trigger 来不及的窗口
- **Kaggle 段位**：800-1070 波动
- **本地 bench 基准**：4P 32%, 2P 47%
- **比赛截止**：2026-06-23 23:59 UTC

## 必读文档（按重要性）

1. `docs/会话总结_本次迭代_2.md` — **最近一次 session 的完整状态 + 下次推荐方向**（每次新会话先读这个）
2. `docs/版本迭代历程.md` — v42 → v100 每版改动 + 教训
3. `docs/Kaggle评分与对手分析.md` — TrueSkill 机制 + 对手分类
4. `docs/对手战术库.md` — 对手风格分类 + named opponent
5. `docs/比赛规则与数据格式.md` — Kaggle 提交硬约束

## 关键事实（防踩坑）

### 提交 vs 训练 的规则边界
- **submission.tar.gz 里的代码（main.py + sim.py + shot_model.py + value_4p_model.py）**：只能 Python 标准库，1s/turn 决策时间
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
- → **结论：v98 在 2P 对 proto1000 的 ~17% 是结构性的，16 个实验全失败**
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
4. 远端 GPU：`zhh@117.50.221.35` (password: `zhh+=123`)，~/orbit_wars_rl/ 有完整 RL pipeline

## 沟通风格

- 用中文回复
- 简洁直接，不要长篇大论
- 给数据 + 给判断，不要无信息客套
- 不确定时直接问，不要装会

## 下次 session 推荐入口

读 `docs/会话总结_本次迭代_2.md` 第五节"下次会话推荐方向"。当前优先：
1. 给 v98 加 coop attack → v105
2. 给 v98 加 proactive reinforcement → v106
3. 写 multi-replay-bot wrapper 用 27 个真实对手当 bench pool
