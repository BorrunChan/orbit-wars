# 分支 plan-b：从胜者 trace 学习的实验

## 思路修正

最早我把对手当纯黑盒——只看 win/lose 来 benchmark。错过了 **env.steps[i][player]['action']** 这个事实：env 已经把每个玩家的 per-turn 决策都存了。

不需要改对手的代码、不需要白盒看他们的算法——直接从 env 输出里抽 (state, action) 即可。

## 流程

1. **[scripts/collect_data.py](../scripts/collect_data.py)**：跑 116 局比赛（12 个 2P 配对 + 17 个 4P 阵容 × 4 seed），把每个胜者的完整轨迹保存到 `data/winner_traces.jsonl`。
2. **[scripts/analyze_winners.py](../scripts/analyze_winners.py)**：聚合胜者的行为统计——发兵时机、规模、目标偏好、阶段差异。
3. **v27 / v28 / v29**：把发现的模式编码进 agent 候选生成。
4. Benchmark 验证：模式有意义但单独应用没显著超过 v20。

## 关键数据（123 个胜者 trace）

| 指标 | proto1000 | lb1224 | structured | 我 v20 |
|---|---|---|---|---|
| **第一次发兵回合** | 3 | **1** | **1** | **12** |
| 每回合动作数 | 4.30 | 2.00 | 2.16 | **0.33** |
| 每回合发兵船数 | 26 | **56** | **48** | 18.6 |
| 中位 fleet size | 17 | 16 | 15 | 36 |
| 派船占源星比例 | 63% | **83%** | **88%** | 异常 |
| 目标平均 prod | 1.0 | **3.0** | **3.0** | 1.0 |

阶段分析（结合 turn 数）也确认：
- **lb1224 / structured 早期 50-70% 回合发射**（我 25%）
- **mid / late 派船比例稳定在 80-90%**（我经常等到能发"全部"才发）

## 实验结果

| 版本 | 改动 | 2P 总 | vs lb1224 | vs proto1000 | vs structured | 4P 强 |
|---|---|---|---|---|---|---|
| v20 (基准) | — | 45% | 60% | 20% | 55% | 5% |
| **v27** | 早期发兵 + 派船变体（half/big）+ prod boost | 42% | 50% | 25% | 50% | — |
| **v28** | 仅早期发兵 override (step<6) | **47%** | 60% | 15% | 65% | **0%** |
| **v29** | 早期 + prod boost | 47% | 60% | 10% | 70% | — |

## 教训

1. **模仿学习的"事实"是对的**（胜者确实早发、多发、找高产），但**直接套规则不够好**。
2. **v20 已经在当前算法骨架的局部最优**附近。加规则只在某些 seed/对手上有效，另一些反而退化。
3. **v28 在 2P 略好（47% vs 45%）但 4P 显著退化（0% vs 5%）**——早期发兵在 4P 桌反而坏事（应该让别人先打）。
4. **个体规则有副作用**：prod boost 让 vs structured 升到 70%，但 vs proto1000 跌到 10%。不同对手要不同的"赢家风格"。

## 没做的（如果继续）

- **regression on (state, action) pairs**：用神经网络/决策树学 state→action 映射。123 trace × 100+ 步 = 1万+ 数据点，理论上够训小模型。
- **训练 game-phase classifier**：早/中/晚期不同的 imitation 策略。
- **针对 4P 单独的模式抽取**：当前数据混了 2P/4P，可能 dilute 信号。

## 当前 plan-b 分支状态

不替换 v20。`main.py` 在 plan-b 分支保留 v20 内容。新东西全在 `agents/v26-v29*.py`、`scripts/collect_data.py`、`scripts/analyze_winners.py`、`scripts/search_b.py`。
