# reyhan-ksatria (Kaggle 1241.8) 策略实现 + 算法说明

> 解耦自 `orbit-wars-reyhan-ksatria.ipynb`,落地于 `external/reyhan/`(`main.py` + `params.json` + `orbit_lite/` + `candidate_metadata.json`)。
> 提交 ref **53704199**,Kaggle **public 1241.8**(75 局收敛)—— **我们目前最高的提交**,较自家 producer 系(expand 1177 / baseline ~1180)**+60~65**。
> 引擎与我们 `single-size`/`exp48` 同源(同一 `orbit_lite`)。算法本体见 `docs/single-size算法详解.md`,本文只讲 reyhan 在其上的**增量**与**为何更强**。

---

## 0. 身份与血脉

- `candidate_metadata.json`:`candidate_id = cand_0615_A_L0014_reinforce_risk`,`family = logistics_regroup`,`created_for = 0615_public_top3_derivatives`,`generator = search/generate_candidates.py`。
- 即:这是一条**自动化参数搜索流水线**从**公开榜前 3 fork** 出来、沿参数轴(此候选改的是 "L" = regroup targets)生成的候选之一。
- **不是新范式** —— 还是 Producer 引擎,赢在**结构小旋钮 + 调好的参数 + 搜索基建**。

---

## 1. = exp48 Producer + 三层增量

reyhan 的 `main.py` = 我们的 `exp48`(Producer:`orbit_lite` 向量化 sim-greedy + flow-diff 打分 + safe_drain + 后勤 regroup + **multi-size 候选** + **endgame 相位**)**加三层**:

### (A) `reinforce_size_beta` —— ETA-aware 增援余量 ⭐ 核心结构增量

在算 capture_floor(捕获一个目标所需的最小兵力)之前,**加一项"目标可能被敌方增援"的余量**,要求发更多兵才打:

```python
beta = config.reinforce_size_beta
enemy_mass = cheap_enemy_pressure(obs, cache, horizon=K_eta, player_id=pid)   # 每星可达敌方质量 [P]
rho = reinforcement_timing_factor(k=1..K_eta, eta_free, eta_scale)            # 反应概率 ramp ∈[0,1]
reinforcement = beta · rho[1,K] · enemy_mass[target][T,1]                     # [T, K] 增援余量
floor = capture_floor(..., reinforcement=reinforcement)                       # 守军 + 余量 再 ceil
```

- 物理意义:**目标越靠近敌方(enemy_mass 大)、飞行越久(rho 随 ETA 升,敌人越有时间反应)→ 要求发越多兵**。
- 作用:**减少"刚好够却被增援打回"的无效/亏本捕获** → 少打亏本仗 → 兵力不被浪费。
- **这是 `orbit_lite` 早就内置但 single-size/exp48 没启用的特性**(`capture_floor(reinforcement=...)` + `reinforcement_timing_factor` 库里都有)。reyhan 把它接上了。
- 相关旋钮:`reinforce_size_beta`(强度)、`reinforce_eta_free=3.0`、`reinforce_eta_scale=12.0`。

### (B) 调好的参数(`params.json`,2P/4P 各一套)

| 旋钮 | 2P | 4P | vs 我们 exp48 |
|---|---|---|---|
| `reinforce_size_beta` | **2.2** | 0.0 | 我们 0(未启用) |
| `min_ships_to_launch` | **3.0** | **3.0** | 我们 4 / 5(更敢发小兵) |
| `size_multipliers` | **[1.0]**(单 size) | **[0.33,0.66,1.0]**(更细) | exp48 [0.5,0.75,1.0] |
| `max_offensive_targets` | 12 | **14** | 我们 7(看更多目标) |
| `max_sources_per_lane` | 12 | **8** | 我们 6 |
| `horizon` | 18 | **15** | 我们 13(更长前瞻) |
| `max_regroup_targets_per_source` | **4** | **4** | 我们 7/8(后勤更收敛) |
| `max_regroup_time` | **5.0** | **5.0** | 我们 7/6(后勤跳更短) |
| `max_waves_per_turn` | 6 | 7 | — |

**取向解读**:**2P 靠 `beta=2.2` 的增援余量 + 单 size + 低 min_ships**(精确、少亏本);**4P 靠 `beta=0` + 细 multi-size(0.33/0.66/1.0)+ 宽目标(14)+ 长 horizon(15)**(广撒、用细 size 控过扩)。后勤层两边都**收紧**(targets=4/time=5,只做短程高价值调度,不浪费)。

### (C) `params.json` 配置覆盖系统(工程价值)

```python
_OW_STRATEGY_PARAMS = json.load(_HERE/"params.json")
CONFIG_2P = _ow_apply_config_overrides(ProducerLiteConfig(), params["config_2p"])
CONFIG_4P = _ow_apply_config_overrides(CONFIG_4P_base, params["config_4p"])
```

`_ow_apply_config_overrides` 只接受 `ProducerLiteConfig` 里存在的字段、`size_multipliers` 转 tuple。**意义:不改代码就能扫参数** —— 这正是搜索流水线的载体。

### endgame 相位(继承自 exp48)

最后 40 回合:`roi 1.5→1.0`、`max_waves→8`、`regroup 关`。

---

## 2. 为什么它能 1241(比我们 +65)

实测(h2h:reyhan base config vs 我们):**reyhan 的配置自然屯出巨型 garrison** —— 一局 2P 跑满 500 步,reyhan 终局 **23 星 / 22428 兵**(对手被压到个位)。即:

- **`reinforce_size_beta` + 低 min_ships + 收紧后勤 + 细 size ⇒ 少打亏本仗、少浪费兵 ⇒ 产能堆成大舰队 ⇒ 长局靠兵力计分碾压。**
- 这与榜首(Isaiah 1790:27 星 / **32117 兵**)同向 —— **屯兵是赢长局的关键**,reyhan 用"调参 + 增援余量"**涌现式**地实现了一部分(不是显式屯兵规则)。
- 对比我们的 exp48/single-size:safe_drain 把星球清空、后勤更松、无增援余量 ⇒ 频繁亏本/浪费 ⇒ 站场兵力低 ⇒ 长局输。这是卡 ~1170 的根因。

**关键教训(已实测)**:显式"到 N 星就抬 roi 停手"的硬屯兵旋钮会 **backfire**(像 v111 冻死扩张:hoard 变体一局只剩 1 星/140 兵)。reyhan 的"屯"是**少浪费**自然涌现的,不是"停手攒兵"。所以方向应是**降低兵力浪费(增援余量、收紧后勤、精确 size)**,而非"到点停止扩张"。

---

## 3. 距榜首(1790)还差什么

reyhan 1241 仍比 Isaiah 1790 低 ~550。调参 + 增援余量拿到 +65,**剩下的大头仍是更彻底的屯兵 / 更优 control**(榜首 32k 兵 vs reyhan 22k vs 我们几百)。下一步杠杆应继续往"减少兵力浪费、把产能更多转成站场兵力"挖,但**避免硬停扩张**。

---

## 4. 运行 / 复现 / 提交

```bash
# 解耦件已在 external/reyhan/。直接提交原版 tarball:
.venv/bin/kaggle competitions submit -c orbit-wars -f external/reyhan/submission_reyhan.tar.gz -m "reyhan 1241"

# 或用参数化引擎(search/agent_main.py 同源,加了 hoard 旋钮)+ 自定义 params.json 打包:
python search/build_submission.py --params external/reyhan/params.json --out external/reyhan/rebuild.tar.gz
```

- 引擎参数化副本:`search/agent_main.py`(= reyhan main + 额外 `hoard_*` 旋钮,默认关)。
- 搜索流水线:`search/search.py`(生成候选 + 本地粗筛;**注意本地对弱池饱和,排名需强对手/Kaggle**)。
- 相关:`docs/single-size算法详解.md`(引擎本体)、`docs/single-size迭代历程.md`(我们的实验史)。
