# 方案 A：从强对手 replay 提取行为规律

## 核心思路

不是真正的 ML 训练，是**行为模式挖掘**：观察 1500+ 选手的 replay，找出他们做什么但我没做的事，作为启发式补丁加进 v20。

**不读他们的代码**（保持黑盒），只看他们在 obs 下产生了什么 action。

## 已有数据

- `Downloads/episode-76456000.json`：bowwowforeach (1700) vs Shun_PI，2P，bowwowforeach 116 步歼灭对手
- 已抽出的早期行为：
  - 第 2 回合就发 13 ships（家里 14）
  - 频繁多 fleet 同回合发送
  - fleet size 跨度大（3 到 200+）

## 需要更多数据

**目标**：30-50 局，至少 5 个不同的 1500+ 选手。涵盖：
- 2P 局（vs 各种对手）
- 4P 局（多人混战）
- 赢的局 + 输的局（看他们在劣势时怎么挣扎）

**怎么拿**：
- 在 Kaggle 选手页面下载历史 episode JSON
- 用 `kaggle competitions episodes <SUBMISSION_ID>` 拿 submission ID 列表
- 用 `kaggle competitions replay <EPISODE_ID>` 拿 JSON
- 或者用户从浏览器下载（已经在做）

## 提取流程

### 第一步：标注每局的关键节点

写脚本 `scripts/analyze_replay.py`，对每个 replay 输出：

```python
{
    "winner": player_idx,
    "first_launch_turn": int,
    "first_launch_size": int,
    "first_launch_ratio": float,    # ships / home_ships
    "avg_fleets_per_turn": float,
    "fleet_size_p50": int,
    "fleet_size_p90": int,
    "target_distance_p50": float,   # 平均派遣距离
    "target_priority": "static" | "rotating" | "mixed",
    "first_capture_turn": int,
    "expansion_rate": [turn → planet_count],  # 时间序列
}
```

### 第二步：聚类分析

把 50 局的提取结果做散点 / 直方图：
- 第一次发兵时刻分布（hypothesis：top 选手都在 turn 1-3 发）
- 首发 fleet 占家底比例（hypothesis：60-100%）
- 平均 fleet 数 / 回合（早期 vs 中期）
- 目标距离 vs 时间（hypothesis：开局打近的，中期打远的）

### 第三步：列出"我没做"的规律

例如可能发现：
- **R1**：开局阶段（step < 30），如果家里 ships ≥ 10，发 80% 至最近未占领星
- **R2**：每回合每个 my_planet 至少发一支（如果有可达目标）
- **R3**：避免送 ships < 5 的小队（速度太慢容易被截）
- **R4**：星球之间力量平衡：把多余船从满星往薄星调

### 第四步：把规律编码成候选生成

不是直接复制规律，是**当作额外候选丢给 sim 评估**：

```python
# 在 v20 的候选列表里追加：
# - "开局攻势"候选：早期发大量船（R1）
# - "饱和发射"候选：每个矿都至少一招（R2）
# - "调度"候选：富星 → 穷星 援军（R4）

# sim 决定要不要采纳
```

这样 **sim 仍然是 gatekeeper**——规律对，sim 接受；规律错，sim 拒绝。

## 预期产出

### v21 候选列表扩充

把 R1-R4 编码为候选生成函数，每回合调用。原 v20 平均生成 30 个 candidates，扩充到 60-80。

`K` 调到 25-30 以纳入这些新候选。

### v22 阶段性策略

如果观察发现强对手**早中晚期策略明显不同**，加阶段切换：

```python
if step < 30:
    # 开局阶段：force-launch 大舰队近距捕获，sim 兜底
    ...
elif step < 200:
    # 中期：v20 的 sim 贪心
    ...
else:
    # 残局：清场，集火残余敌方星
    ...
```

只在数据**真显示**强对手有这种切换时才加，不要拍脑袋。

## 风险

- **样本偏差**：不同选手风格不一样，"平均"出来可能没人这么打
- **过拟合**：把规律编死了，遇到未见的对手类型反而僵化
- **不可解释行为**：top 选手做的最关键决策可能根本无法从 replay 看出（涉及他们的内部 sim 状态、上一回合的判断等）
- **数据收集慢**：每局 replay JSON 是 1-3 MB，30 局就是 50+ MB，手动下载累

## 工作量估计

| 步骤 | 时间 |
|---|---|
| 用户下载 30 局 replay | 1-2 h |
| 写 `analyze_replay.py` 抽统计 | 1 h |
| 看结果 + 列规律 | 1-2 h |
| 编码 R1-R4 候选 | 2-3 h |
| benchmark 调优 | 2-3 h |
| **总计** | **8-12 h** |

## 上限估计

乐观估计提升：**45% → 60-65% (vs 强对手)**。

天花板原因：top 选手的复杂搜索结构（sim horizon 110、mission 系统、深度 lookahead）我没法仅靠规律复现。

## 不做白盒的边界

- ✅ **允许**：分析他们在公开 replay 里产生的 action，归纳模式
- ✅ **允许**：参考他们用的概念（深 sim、阶段策略、协调任务）——这些是 RTS 通用知识
- ❌ **不做**：导入 `lb1224.py` 当作 sim 里的对手模型
- ❌ **不做**：把他们的代码片段直接复制进我的 agent
