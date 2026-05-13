# V(state) 价值函数实验失败复盘

模仿 `lb-highest-1000-search-learned-value-function` 的思路：训一个 GBC 学 V(state) → P(win)，替换我 sim 的 evaluate。**4 次实验全部 ≤ v20，最佳一次只是误差内**。

## 实验汇总

| 配置 | 训练数据 | AUC | 5-seed vs 强对手 | 相比 v20 |
|---|---|---|---|---|
| v20 基线（无 V）| — | — | 45% | 0 |
| 100T × d4 | 65 局 partial | 0.971 | **50%** | +5pp |
| 200T × d5 | 228 局 | 0.972 | **17%** | -28pp 崩盘 |
| 100T × d3 | 318 局 | 0.960 | **30%** | -15pp |
| sim_eval + 0.2 × V（混合） | 318 局 | — | **27%** | -18pp |

**第一次 50% 是运气**——更多数据 + 更复杂模型反而显著退化。

## 根本原因：训练/部署分布漂移

V 学的是 **真实游戏 state → 最终胜者**。GBC 内部把训练集里 state 特征跟胜者标签的相关性学到。

但我在 agent 里用 V 评分 **sim 推演出来的 state**：
- 我把 obs 喂进 sim
- sim 把状态向前推 16 步（用 starter 模型估计对手）
- 用 V 给推演结果打分

**sim 状态 ≠ 真实状态**：
- 我假设对手用 starter 风格走每一回合
- 真实对手是 lb1224/proto1000/structured 之类，行为完全不同
- 我的 sim 不模拟新彗星生成
- 16 步推演里复杂动态的累积误差很大

V 看到一个"不像真实游戏的怪状态"，给出的 P(win) 可能完全离谱。**模型越复杂、对训练分布越敏感、漂移就越致命**。

## 为什么 lb-highest-1000 能做出来

那篇 notebook 的作者明确写过：
> Self-play with v11 itself rather than heuristic (closes the distribution mismatch — the value function was trained on real-game states but applied to simulated states)

他们的训练数据里 **240,982 个 (state, winner) pair 来自 1000 局 self-play**——也就是说他们让自己的 sim agent 互打 1000 局生成训练数据。这样 V 学的就是"sim 视角的 state 分布"，部署时分布一致。

我只用了真实对战的 330 局 = ~127k pair，**没有 self-play 数据**。距离他们的设置差 8x 数据量 + 关键的 sim 分布。

## 如果继续推

要让 V 真正工作需要：
1. **用 sim 跑 self-play 生成训练数据**（v20 vs v20，几百局，记录 sim 视角 state + 最终胜者）
2. 训 V 时同时用 real-game data 和 self-play data
3. 重新评估

这是 1-2 天 CPU 时间 + 工程量，没在这次迭代窗口完成。

## 当前决定

回退 v20，保持 Kaggle 上原提交。

- `main.py` = v20（参数搜索得来的最佳配置）
- `submission.tar.gz` 重新打包（main.py + sim.py）
- V 训练 pipeline 和数据集留在 `data/` 和 `agents/v30_value.py`，将来如果要做 self-play V 重做

## 数据资产保留

`data/round_robin_games.jsonl`：330 局 11 个 agent 互打的完整数据（25MB+）。哪怕这次 V 失败，这数据还有用：
- 可以做地图聚类分析（哪些对手在哪些地图上赢得多）
- 可以做更精细的特征工程
- 可以做 imitation learning 的对照实验

## 教训

1. **高 AUC 不等于好策略**：AUC 测的是"分得开胜/负 state"，不测"决策正确"。
2. **训练分布 vs 部署分布**——这是 ML 系统典型坑，sim-based agent 尤其敏感。
3. **不要相信单次小样本结果**：第一次 50% 看起来香，但只是 30 局的运气。复现需要看是否在更多 seed 下稳定。
4. **Simpler is safer**：手调参数（v20）至少在已知对手上稳定 45%。复杂 ML pipeline 没收益就是负收益。
