# v36 多维地图聚类实验 — 失败

## 思路

v35 用 `angular_velocity` 一维做地图分类，只有 2 个 bucket。我想：用 **10 维特征**聚类成 4 个更精细的 cluster，对每个 cluster 用最适合的参数预设。

## 实施

1. 收集 **30 个 unique seeds × 12 个 agent pairings = 360 局** 数据（[scripts/collect_diverse.py](../scripts/collect_diverse.py)）
2. 10 维特征聚类（K-means K=4）（[scripts/cluster_maps.py](../scripts/cluster_maps.py)）
3. 每 cluster 分析胜者
4. 写每 cluster 的 shot_threshold preset
5. [agents/v36_cluster_dispatch.py](../agents/v36_cluster_dispatch.py) + `cluster_params.json` 在 runtime 根据 10 维特征向量分类

## 聚类结果（30 seeds, 360 games）

| Cluster | 地图特征 | mlhybrid | **main** | lb1224 | structured |
|---|---|---|---|---|---|
| **0** | 中快转 + 中产 + 远家 | **87%** | **17%** | 50% | 47% |
| **1** | 中快 + 产能均匀 | **75%** | **25%** | 46% | 54% |
| **2** | 慢-中 + 高产 + 远家 | 62% | **79%** ⭐ | 33% | 25% |
| **3** | 中速 + 产能多样 + 多星 | **82%** | 55% | 31% | 32% |

理论上限：如果每 cluster 我都能匹配该 cluster 最强者，胜率从 51% 跳到 81%。

## Benchmark 结果

| 版本 | mlhybrid | orbitbot | proto | struct | lb1224 | 总 |
|---|---|---|---|---|---|---|
| **v35**（av-only） | 30% | 20% | 20% | 50% | **60%** | **36%** |
| v36（4 cluster, 4 阈值）| 50% | **0%** | 20% | 40% | 30% | 28% |
| v36b（hybrid: av + cluster 0/2 override）| 50% | **0%** | 20% | 40% | 40% | 30% |

**v36/v36b 净退化**——vs mlhybrid +20pp 但 vs lb1224 / orbitbotnext 大跌。

## 为什么失败

**调阈值不等于换算法**。

对 cluster 0 地图（mlhybrid 87% 王国），我把阈值从 0.40 调到 0.50（更接近 mlhybrid 的严格），但**我 v35 的底层算法不是 mlhybrid 的 architecture**。

mlhybrid 的 ML rejecter 是基于完整的不同 architecture（规则 + 完整 ML 验证），不只是"阈值高"。我只调一个数字模仿不了人家。

具体证据：仅仅 seed 2（cluster 0）从 0.40 → 0.50 一个改动，就让 vs orbitbotnext 从 20% 跌到 0%。说明 0.50 阈值对我的 v20 base 是**严重不适合**，不像对 mlhybrid 那样有效。

## 实际教训

1. **数据驱动的 cluster 分类是对的方向**：清晰看到 4 类地图、4 类胜者分布
2. **"换参数应对地图" 只在小范围内有效**：v35 的简单二分（av < 0.035）是这个机制的甜点
3. **"换算法应对地图" 才是真正的解决**：要在 cluster 0 / 1 上突破，需要写**完全不同的策略实现**（不是参数调优）

## 实际下一步（如果继续）

要让多 cluster dispatch 真的有效，需要为每个 cluster 写**真正不同的算法**：

```python
if cluster == 0:
    return mlhybrid_style_agent(obs)   # 完整重写，不只是参数
elif cluster == 2:
    return v20_style_agent(obs)
...
```

工程量大概是当前 main.py 的 3-4 倍代码。不是这轮迭代窗口能完成的。

## 当前提交

回退 **v35**。`submission.tar.gz` 含 `main.py` (v35) + `sim.py` + `shot_model.py`。
v35 在 5 强对手 50 局下平均 36%，是当前算法骨架的最优配置。

数据资产（gitignored）：
- `data/diverse_games.jsonl`：30 seeds × 12 pairings = 360 局含 map_features
- `data/map_clusters.json`：聚类结果 + 每 cluster 胜者统计
