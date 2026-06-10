# 任务交接：Orbit Wars 学习型 value self-play league（给 3060 机的 agent）

你是另一台带 **8GB RTX 3060 (CUDA)** 笔记本上的 coding agent。本文档让你**立即开干**：在本机训练一个学习型 value 函数，用 self-play league 迭代，产出可提交 Kaggle 的 agent。所有脚手架代码已在仓库 `rl/` 下、已本地冒烟通过。你的工作 = **跑迭代循环、判断收敛、导出、（回报结果）**。

---

## 0. 背景（为什么做这个 / 别踩的坑）

- 主线 agent 是 **Producer**（torch 向量化 sim-greedy，`single-size/main_exp48.py`）。所有手工变体（multi-size/endgame/react/expand/catchup）**Kaggle 全挤在 ~1160–1180**，封顶。
- 诊断：封顶根因是 **control/execution**（赢"产能竞赛"），不是 value estimation。失败模式是**中期~4 星平台**（25–70% 进度停止扩张被对手雪球碾死）+ **对屯兵者（NMarkS：星数打平、屯兵 11×、拖满 500 步靠终局兵力计分赢）无解**。
- 方向 ③ = **学一个 value(state)→P(win)，用它在 producer 的候选上做 1-ply value-search**（保留 producer 靠谱的候选生成，只换短视的 flow-diff 评估）。
- ⚠️ **必须避开 v82 的坑**：上一次 RL（PPO/BC 整 policy）**不外推到 Kaggle 真实场**。本设计的防御：
  1. **只学 value、不学 policy**（搜索那层是手写的，鲁棒）。
  2. **league/多样对手池**（producer 各变体 + structured/proto1000/orbitbotnext + 过去的 value 快照），**绝不纯自博弈对单一对手**。
- ⚠️ **Kaggle 分噪声**：同代码波动 ±~100，**~15 局才收敛**；**别看单次分**，要多局/持续。本地 head-to-head 胜率是更快的迭代信号（但仍非最终裁判）。

---

## 1. 环境搭建

```bash
git clone <这个仓库>            # 或同步 bor 的 orbit-wars 仓库
cd orbit-wars
python -m venv .venv && source .venv/bin/activate    # 或用 conda
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.1 轮子；按你的 CUDA 版本调
pip install kaggle-environments numpy
# 验证 GPU：
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

需要的目录已在仓库里：`single-size/orbit_lite/`（引擎包）、`single-size/main_exp48.py`（base producer）、`opponents/*.py`（对手池）、`rl/`（本 pipeline）。

冒烟自检（CPU 也能过，~1 分钟）：
```bash
python rl/gen_selfplay.py --games 6 --out rl/data/smoke.jsonl --workers 3
python rl/train_value.py --data rl/data/smoke.jsonl --out rl/data/value_smoke.json --epochs 20
# 期望：holdout_AUC 打印出来 + 文件生成
```

---

## 2. 架构（已实现，按需读代码）

| 文件 | 作用 |
|---|---|
| `rl/features.py` | 单帧 obs→特征向量（**训练/推理共用**，纯 python，FEATURE_ORDER 勿改否则权重失效） |
| `rl/value_net.py` | 小 MLP（torch 训练）+ JSON 权重导出 + **纯 python forward**（部署用，1s/turn 安全） |
| `rl/gen_selfplay.py` | 多进程跑 league 对局，采样 (特征, 是否 rank1) → jsonl |
| `rl/train_value.py` | 训练 value（CUDA 自动），输出 JSON 权重 + holdout AUC |
| `rl/value_agent.py` | **部署 agent**：exp48 候选 + 在 3 个激进度配置(roi 1.0/1.5/2.5)间，用 V 评估"投影到 horizon 的结果态"择优。**无权重时自动退化为纯 exp48（保证不劣于基线）** |
| `rl/run_league.py` | 编排：gen→train→eval(value vs exp48 head-to-head)→快照，循环 |
| `rl/build_submission.py` | 打包 submission.tar.gz |

value-agent 每回合 ~45–60ms（远低于 1s）。已本地验证 2P/4P 跑通、解压独立运行。

---

## 3. 主任务：跑 league 迭代循环

```bash
# 一条命令跑 6 轮迭代（每轮 300 局 self-play + 训练 + 评估）。workers 设成 CPU 核数-1。
python rl/run_league.py --iters 6 --games_per_iter 300 --workers 6
```

每轮会打印：
```
=== iter N: value vs exp48 head-to-head = XX% over G games ===
```

**判定（这是你的核心工作）：**
- **head-to-head 胜率 > 50% 且随迭代上升** → value-search 在赢过 base producer，方向成立，继续迭代直到平台。
- **始终 ≈ 50%（在噪声内）** → value 没带来真实 control 增益（多半 value 和 flow-diff 冗余，呼应离线发现 prod_share AUC 0.98）。**这时不要硬训**，回报"无增益"，停。
- **< 50%** → value-search 有害（可能 over-conservative），调 `_VARIANTS`（`rl/value_agent.py` 里的 roi 档位）或 features 后重试一轮；仍不行则停。

GPU 在这里只加速 `train_value`（几秒~分钟）；**瓶颈是 CPU 跑对局**（gen_selfplay 多进程）。3060 笔记本核少的话，每轮 300 局可能 10–30 分钟，按需调 `--games_per_iter`。

> 想更快迭代/更多局：把 `--games_per_iter` 调大、`--iters` 调大；数据是累积训练的（train 吃所有 iter*.jsonl）。

---

## 4. 导出 + 提交（迭代收敛后）

```bash
# 选 head-to-head 最高的那轮权重（通常 rl/data/value_latest.json = 最后一轮）
python rl/build_submission.py --weights rl/data/value_latest.json --out rl/submission_rl.tar.gz
# 验证解压独立运行（重要）：
mkdir -p /tmp/chk && tar xzf rl/submission_rl.tar.gz -C /tmp/chk
python scripts/smoke_test.py /tmp/chk/main.py     # 期望 PASS
```

提交（bor 的 Kaggle CLI 已认证，`.venv/bin/kaggle`；若你这台没认证，把 tarball 回传给 bor 提交）：
```bash
.venv/bin/kaggle competitions submit -c orbit-wars -f rl/submission_rl.tar.gz \
  -m "rl value-search: learned value re-ranks exp48 aggression variants"
```

**提交后**：等 ~15+ 局收敛，`kaggle competitions submissions -c orbit-wars` 看分，对照 base ~1180。**别看前几局的低分**（新提交会从低位爬，参考历史：react 600→1211）。

---

## 5. 停止 / 回报标准

- **成功**：value-agent head-to-head 持续 >55% 打过 exp48 **且** Kaggle 收敛分 > 1180（多局）。→ 这就是新主力，回报权重文件 + 分数。
- **失败/无增益**：head-to-head ≈50% 或 Kaggle 收敛 ≤1180。→ 回报"③ value-search 无穿透噪声的增益，封顶 ~1180 成立"，**别再砸时间**。
- 任何崩溃/超时（>1s/turn）：贴报错；value_agent 有无权重 fallback，先确认 fallback 正常。

回报内容：每轮 head-to-head 胜率曲线、最终 Kaggle 分（多次读数）、权重文件 `rl/data/value_latest.json`、遇到的问题。

---

## 6. 可调点（若 head-to-head 停在 ~50%，按序试）

1. `rl/value_agent.py` `_VARIANTS`：增加/调整激进度档位（roi、max_waves），给 value 更多可选动作。
2. `rl/features.py`：加更强特征（如 per-opponent 兵力分布、前线压力、在途舰队、距 500 步剩余）。**注意改了要重训，且部署权重必须配套**。
3. `rl/value_agent.py` 投影：现在用 do-nothing garrison_status 投影结果态。可换更长 horizon 或加一步对手反应注入（参考 `single-size/main_react.py` 的 `_predict_opponent_arrivals`）。
4. league 对手池（`gen_selfplay.py` DEFAULT_POOL）：加更多样对手防过拟合。

核心原则不变：**只学 value、league 多样化、别看单次分、本地 head-to-head 当快速 gate、Kaggle 多局当最终裁判。**
