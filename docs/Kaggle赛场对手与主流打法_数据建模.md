# Kaggle 赛场对手与主流打法 — 数据建模

> 数据驱动,基于 **998 局真实 Kaggle 回放**(CLI 拉取全部 14 个 Borrun submission 的 episodes)+ **2993 队实时 leaderboard 评分** join。生成日期 2026-05-20。
> 脚本:`scripts/model_opponents.py`。数据:`data/opponent_styles.{jsonl,md}`。
> 这份文档的目的:**判断真实赛场的主流打法、哪种风格真赢、我们差在哪。**

---

## 1. 四大 archetype 分布(1028 个有评分对手)

按 launch_rate 自动分类:

| archetype | launch/t | n_opps | mean_lb | median_lb |
|---|---|---:|---:|---:|
| high_volume_swarm | ≥3.0 | 97 | 837 | 864 |
| moderate_pressure | 1.2-3.0 | 222 | 810 | 812 |
| precise_structured | 0.3-1.2 | 510 | 813 | 788 |
| passive_turtle | <0.3 | 199 | 794 | 766 |

**4 类的 mean_lb 几乎一样(794-837)。"选哪个风格"不决定输赢。** 这是第一个反直觉结论。

---

## 2. ⭐ 什么真正与高评分相关(420 个 ≥2 局对手)

| 特征 | Pearson r vs lb_score | 解读 |
|---|---:|---|
| **launch_rate** | **-0.017** | **零相关 — 高 volume 不是 meta** |
| avg_fleet | +0.271 | 最强,但**疑似反向因果**(赢→兵多→舰队大) |
| planets@25(早期/最接近因果) | +0.157 | 早期扩张有用 |
| planets@50 | +0.026 | 弱 |
| peak_planets | -0.044 | 零/负 |

**关键**:即使最强的 avg_fleet(r=0.27)也只解释 ~7% 方差;早期扩张 planets@25(r=0.16,最不受反向因果污染)只解释 ~2.5%。
**→ 没有任何粗粒度特征能解释评分。决定胜负的是 aggregate 看不见的执行质量(目标选择、防御、时机、守住占领、地图自适应)。**

---

## 3. 榜首到底什么风格(Top-50 rated opponents)

| archetype | 占比 |
|---|---:|
| **precise_structured** | **28/50 = 56%** |
| passive_turtle | 11/50 = 22% |
| moderate_pressure | 7/50 |
| high_volume_swarm | 4/50 = 8% |

Top-50 平均:**launch_rate 0.93(低)** / **avg_fleet 49.4(大)**。

榜首样本:
| rank | opp | lb | launch/t | fleet |
|---:|---|---:|---:|---:|
| 1 | Vadasz | 1636 | 0.65 | 42.5 |
| 2 | bowwowforeach | 1594 | 0.64 | 44.0 |
| 6 | Isaiah@Tufa | 1534 | 0.51 | 64.6 |
| 12 | Shun_PI | 1443 | 0.66 | **126** |

**主流赢家打法 = 精确 + 低 launch(~0.9) + 大舰队(~49) + 强早期扩张。不是高 volume。**

---

## 4. high_volume 是 mid-table 陷阱

high_volume 类(25 个 ≥2 局):
```
lb 分布: min 472 | p25 835 | median 876 | p75 949 | max 1428
> 1000: 4/25     > 1300: 1/25
```

高 volume 能到 mid-high(median 876),**但几乎到不了顶(1/25 上 1300)**。它打得赢 v124(我们 0/23),却撞在 ~900 天花板。

---

## 5. Borrun 自己在地图上的位置(835 局)

| | launch/t | fleet | planets@50 | peak | lb |
|---|---:|---:|---:|---:|---:|
| TOP-50 对手 | 0.93 | 49.4 | 6.3 | 16.7 | 1300+ |
| **Borrun** | **0.45** | **41.1** | **6.8** | **17.3** | **776** |
| BOTTOM 25% | 1.07 | 20.8 | 5.1 | 12.4 | 612 |

**悖论**:我们的 **fleet(41,接近 top)和扩张(peak 17.3,超过 top)都已达标**,唯一异常是 launch_rate 0.45(比谁都低,是 top 的一半)。但 launch_rate 零相关——所以"launch 低"理论上不该让我们卡 776。

**悖论的解**:我们已经在打"赢家 archetype"(precise + 大舰队 + 好扩张),却只有 776。说明 **776→1636 的差距不在风格、不在任何粗粒度特征,而在执行深度**——目标选择质量、是否守得住占领、防御反应、时机、地图自适应。这些 aggregate 测不出来,也正是 v98→v124 17 个调参实验全失败的根因(他们在调风格旋钮,但旋钮不是瓶颈)。

---

## 6. 对策略的硬结论

### 6.1 v128(高 volume)方向需要重新评估
- 高 volume median 876 > 我们 776 → 部署 v128 **可能** +100 到 ~840-876
- **但它是已知的死路天花板**(1/25 上 1300)。我们当前 archetype(precise)天花板是 1636
- **我们是"高天花板里的弱执行者(776)",不是"选错了 archetype"**。换成 high_volume = 从高天花板跳到低天花板
- **判断**:v128 顶多是一次廉价的 floor-test(可能小涨),不是通往榜首的路。别在它上面重投入

### 6.2 真正的路:在 precise 范式内提升执行
我们已经有大舰队 + 好扩张。缺的是**看不见的执行质量**:
- 守住占领(你之前的"占领后被夺回"分析 — 大舰队我们有了,但可能投放时机/目标错)
- 防御反应(proto1000 有 reinforcement,我们没有)
- 目标选择质量
这些**无法用本地 bench 或 aggregate 验证**,只能靠 Kaggle 真实信号迭代

### 6.3 唯一可信的验证回路
- 本地 standard pool:反相关,已证明(v98→v124 本地优化 = Kaggle 负收益)
- aggregate 特征:解释力 <7%,不能当优化目标
- **只有 Kaggle 实测 + 本文档这套回放分析回路是真信号**

---

## 7. 数据资产 + 复现

```
data/opponent_styles.jsonl   1030 对手 × {launch_rate, avg_fleet, expansion, archetype, lb_score, lb_rank}
data/opponent_styles.md      自动生成的 archetype/相关性/top 表
scripts/model_opponents.py   重建脚本(读 replays/ + leaderboard CSV)
replays/{2p,4p}/             998 真实回放(CLI 已拉全)
```

更新数据(部署新版打了新局后):
```bash
# 1. 拉新回放
.venv/bin/kaggle competitions submissions orbit-wars   # 取新 submission_id
.venv/bin/kaggle competitions episodes <id>            # 列 episodes
.venv/bin/kaggle competitions replay <ep> -p replays/  # 下载
# 2. 重建模型
.venv/bin/python scripts/model_opponents.py
```

---

## 8. 一句话总结

**真实 Kaggle meta = 精确流 + 大舰队 + 强早期扩张 + 高执行质量,绝不是高 volume。我们已经在打这个 archetype 且 fleet/扩张达标,卡在 776 是执行深度问题,不是风格问题。没有可调的"魔法旋钮"——这正是为什么 17 个调参实验全失败。前路只能靠 Kaggle 真实信号迭代执行细节。**
