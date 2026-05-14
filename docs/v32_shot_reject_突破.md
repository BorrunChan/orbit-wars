# v32 Shot-Reject 突破 mlhybrid 困局

## 问题

`v20` 对 **mlhybrid** 完败（0/10）。Kaggle 上很多 1000+ bot 都是 mlhybrid 系（"规则候选 + ML 拒坏发射"流派），所以这是结构性弱点。

## 思路：复刻 mlhybrid 的核心机制

mlhybrid 的 ML 模型作用是 **否决可能失败的发射**——一种学习到的 "shot validator"。

我做了：
1. **数据收集**：v20 vs 7 个强对手 × 3 seeds × 2 sides = 42 局（[scripts/collect_shots.py](../scripts/collect_shots.py)）
2. **抽 shots**：每个动作 = 一次 shot，用 ray-cast 识别目标，前看 20 回合判定是否成功捕获（[scripts/extract_shots_v2.py](../scripts/extract_shots_v2.py)）
3. **训分类器**：6272 个 shot，67.9% 成功率。GBC 200T × d4，AUC 0.918（[scripts/train_shot.py](../scripts/train_shot.py)）
4. **集成**：v32 在候选生成后用分类器过滤，P(success) < 0.40 的拒绝（[agents/v32_shotreject.py](../agents/v32_shotreject.py)）

## 训练特征（按重要性）

```
distance              0.169  ← 距离
margin                0.161  ← ships - predicted_garrison
my_planets_count      0.159  ← 全局态势
my_total_ships        0.140
tgt_ships             0.089
opp_total_ships       0.071
predicted_garrison    0.050
eta                   0.040
fleet_speed           0.025
step_progress         0.020
```

距离 + margin + my_planets_count 占主导——基本上是"边远小队 / 兵力差距 / 我方实力"的综合判断。

## Benchmark

每对 5 seeds × 2 sides = 10 局：

| 对手 | v20 (无过滤) | **v32 (P > 0.40 过滤)** | Δ |
|---|---|---|---|
| **mlhybrid** | 0% | **30%** | **+30** ⭐ |
| **orbitbotnext** | 20% | **40%** | **+20** ⭐ |
| proto1000 | 30% | 20% | -10 |
| structured | 50% | 30% | -20 |
| lb1224 | 60% | 50% | -10 |
| **总 (5 强对手)** | **32%** | **34%** | **+2** |
| vs random | 100% | 100% | 0 |
| vs starter | 100% | 100% | 0 |
| vs v2d_conserve | 100% | 100% | 0 |

**关键观察**：
- 总分提升小 (+2pp)，但**专门补了 v20 最弱的两个对手**
- 弱对手 100% 不变（filter 不会误伤普通捕获）
- 中等强对手 (proto/struct/lb1224) 有 10-20pp 小退化——P=0.40 阈值对它们偏保守

## 其他试过的配置

| 配置 | mlhybrid | orbitbot | proto | struct | lb1224 | 总 |
|---|---|---|---|---|---|---|
| v32 t=0.40（**最终选择**）| **30** | **40** | 20 | 30 | 50 | **34** |
| v32 t=0.30 | 30 | 40 | 20 | 50 | **20** | 32 |
| v32 软 blend (score × √P) | 10 | 0 | 20 | **80** | 50 | 32 |

每个配置改变胜负分布但总量稳定 ~32-34%。**t=0.40 是最稳的 trade-off**——补最大短板，其他小退化可接受。

## Kaggle 预期

Kaggle 上 mlhybrid 派比例如果接近 30-50%，**v32 应该比 v20 涨 50-100 ELO**：
- 之前 v20 vs mlhybrid 派 = 全输 → ELO 下行
- v32 vs mlhybrid 派 = 30% 胜 → 不再单边崩

## 提交

`submission.tar.gz` (60KB) 含：
- `main.py` (v32_shotreject, 16KB)
- `sim.py` (9KB)
- `shot_model.py` (200KB, GBC 树 dump)
