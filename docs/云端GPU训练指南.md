# 云端 GPU 训练指南

完整指南：把 4P value function 训练放云上，模型拿回本地用。

## 目标

把这个流程跑通：
```
本地数据 (v4p_games.jsonl)
       ↓ 上传
   云 GPU (Colab / Kaggle / 自己的服务器)
       ↓ 运行 train_value_4p.py --gpu
   value_4p_model.py (~500KB-2MB)
       ↓ 下载
本地 main.py + sim.py + shot_model.py + value_4p_model.py → submission.tar.gz
```

---

## 方案一：Google Colab（最推荐）

### 优点
- **免费**（T4 GPU，每天约 8 小时）
- 不用装任何东西
- 完全浏览器操作

### 步骤

**1. 准备数据**

本地数据收集跑完后（约 5-8 小时），会有 `data/v4p_games.jsonl`。

确认它在那里：
```bash
ls -lah /Users/bor/Projects/orbit-wars/data/v4p_games.jsonl
```

应该看到几十 MB。

**2. 打开 Colab**

去 https://colab.research.google.com → "New notebook"

**3. 切到 GPU 运行时**

菜单 → Runtime → Change runtime type → Hardware accelerator: **T4 GPU** → Save

**4. 上传数据**

点左侧文件夹图标 → "Upload" → 选 `data/v4p_games.jsonl`

或者用代码上传（粘到 cell 1）：
```python
from google.colab import files
uploaded = files.upload()  # 弹文件选择框
```

**5. 准备训练脚本**

在 cell 2 粘整个训练脚本（我等下提供单文件版）。

**6. 跑训练**

```python
!python train_value_4p_inline.py --gpu --trees 800 --depth 7
```

应该几分钟内完成。看到：
```
Validation AUC: 0.93XX
File size: XXX,XXX bytes
```

**7. 下载结果**

```python
files.download('value_4p_model.py')
```

浏览器自动下载到本地 Downloads。

**8. 放回项目**

```bash
mv ~/Downloads/value_4p_model.py /Users/bor/Projects/orbit-wars/value_4p_model.py
```

---

## 方案二：Kaggle Notebooks（也免费）

差不多流程，但你已经在 Kaggle 有账号了。

1. https://www.kaggle.com → Code → New Notebook
2. Settings → Accelerator: **GPU T4 x2** （免费，每周 30 小时）
3. Upload `v4p_games.jsonl` 到 notebook（Add Data → Upload）
4. 粘训练脚本 + 跑
5. 右下载 output 区找 `value_4p_model.py`

---

## 方案三：自己的 VM（AWS/GCP/Lambda Labs 等）

如果你已有 SSH 进 GPU 机器：

```bash
# 1. 上传数据 + 脚本（本地终端）
scp data/v4p_games.jsonl user@gpu-server:/path/to/work/
scp scripts/train_value_4p.py user@gpu-server:/path/to/work/

# 2. SSH 进去（gpu-server 终端）
ssh user@gpu-server
cd /path/to/work

# 3. 装依赖
pip install xgboost numpy scikit-learn

# 4. 跑训练
python train_value_4p.py --gpu --trees 800 --depth 7

# 5. 拿回模型（本地终端）
scp user@gpu-server:/path/to/work/value_4p_model.py ./
```

---

## Colab 训练 cell（自包含）

下面整段贴到 Colab notebook 一个 cell 里。前提：已上传 `v4p_games.jsonl`。

```python
# Cell: install + train + download
!pip install xgboost==2.0.3 -q

import json, math, sys
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss

FEATURE_NAMES = [
    "step_progress", "is_2p", "is_4p",
    "my_ship_frac", "my_planet_frac", "my_prod_frac", "my_centrality",
    "max_opp_ship_frac", "max_opp_planet_frac", "max_opp_prod_frac",
    "opp_count_alive", "min_enemy_dist_norm",
    "ship_lead", "planet_lead", "prod_lead", "my_fleet_frac",
]

def build_features_4p(state_dict, num_players):
    step = state_dict["step"]
    my = state_dict["my"]; opps = state_dict["opps"]
    my_ships = my["ships"]; my_planets = my["planets"]
    my_prod = my["prod"]; my_fleet_ships = my["fleet_ships"]
    my_centrality = my["centrality"]
    min_enemy_dist = state_dict.get("min_enemy_dist", 100.0)
    opp_ships = sum(o["ships"] for o in opps)
    opp_planets = sum(o["planets"] for o in opps)
    opp_prod = sum(o["prod"] for o in opps)
    opp_fleet_ships = sum(o["fleet_ships"] for o in opps)
    max_opp_ships = max((o["ships"] for o in opps), default=0)
    max_opp_planets = max((o["planets"] for o in opps), default=0)
    max_opp_prod = max((o["prod"] for o in opps), default=0)
    opp_count_alive = sum(1 for o in opps if o["planets"] > 0 or o["ships"] > 0)
    total_ships = max(1, my_ships + opp_ships)
    total_planets = max(1, my_planets + opp_planets + state_dict.get("neutral_planets", 0))
    total_prod = max(1, my_prod + opp_prod)
    total_fleet_ships = max(1, my_fleet_ships + opp_fleet_ships)
    sd = lambda a, b: a / b if b > 1e-9 else 0.0
    return [
        step / 500.0,
        1.0 if num_players == 2 else 0.0,
        1.0 if num_players == 4 else 0.0,
        sd(my_ships, total_ships),
        sd(my_planets, total_planets),
        sd(my_prod, total_prod),
        my_centrality / 60.0,
        sd(max_opp_ships, total_ships),
        sd(max_opp_planets, total_planets),
        sd(max_opp_prod, total_prod),
        opp_count_alive / 3.0,
        min(1.0, min_enemy_dist / 100.0),
        sd(my_ships - opp_ships, total_ships),
        sd(my_planets - opp_planets, total_planets),
        sd(my_prod - opp_prod, total_prod),
        sd(my_fleet_ships, total_fleet_ships),
    ]

# Load data
X, y = [], []
n_games = 0
with open("v4p_games.jsonl") as f:
    for line in f:
        g = json.loads(line)
        n_games += 1
        winner = g["winner"]
        num_players = g.get("num_players", 4)
        for p_idx_str, traj in g["traces"].items():
            p_idx = int(p_idx_str)
            label = 1 if p_idx == winner else 0
            for turn in traj:
                X.append(build_features_4p(turn["state"], num_players))
                y.append(label)
print(f"Loaded {n_games} games → {len(X)} samples")

X = np.array(X, dtype=np.float32); y = np.array(y, dtype=np.int32)
print(f"Positive rate: {y.mean():.1%}")

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)

params = {
    "objective": "binary:logistic", "eval_metric": "logloss",
    "max_depth": 7, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85,
    "tree_method": "hist", "device": "cuda",
}
dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES)
dtest = xgb.DMatrix(X_te, label=y_te, feature_names=FEATURE_NAMES)
print("Training on GPU...")
booster = xgb.train(params, dtrain, num_boost_round=800,
                     evals=[(dtest, "test")],
                     early_stopping_rounds=50, verbose_eval=100)
p_te = booster.predict(dtest)
print(f"\nValidation AUC: {roc_auc_score(y_te, p_te):.4f}")
print(f"Validation Brier: {brier_score_loss(y_te, p_te):.4f}")

# Dump pure-Python trees
trees_data = booster.get_dump(dump_format='json')
out_lines = [
    '"""Auto-generated V(state) classifier (4P)."""',
    f'V_BASE = 0.0',  # XGB uses sigmoid(0) = 0.5 as default
    f'V_N_TREES = {len(trees_data)}',
    'V_TREES = [',
]
for tree_json in trees_data:
    tree = json.loads(tree_json)
    nodes = []
    def flatten(node):
        if 'leaf' in node:
            nodes.append({"feat": -2, "thr": 0.0, "val": node["leaf"],
                           "left": -1, "right": -1}); return len(nodes) - 1
        cur_idx = len(nodes); nodes.append(None)
        feat_str = node["split"]
        feat = int(feat_str[1:]) if feat_str.startswith("f") else FEATURE_NAMES.index(feat_str)
        thr = node["split_condition"]
        children_by_id = {c["nodeid"]: c for c in node["children"]}
        left_idx = flatten(children_by_id[node["yes"]])
        right_idx = flatten(children_by_id[node["no"]])
        nodes[cur_idx] = {"feat": feat, "thr": thr, "val": 0,
                          "left": left_idx, "right": right_idx}
        return cur_idx
    flatten(tree)
    feat_list = [n["feat"] for n in nodes]
    thr_list = [round(n["thr"], 6) for n in nodes]
    val_list = [round(n["val"], 8) for n in nodes]
    left_list = [n["left"] for n in nodes]
    right_list = [n["right"] for n in nodes]
    out_lines.append(f'    ({feat_list}, {thr_list}, {val_list}, {left_list}, {right_list}),')
out_lines.append(']')
out_lines.append('')
out_lines.append('import math as _math')
out_lines.append('def value_predict(feats):')
out_lines.append('    z = V_BASE')
out_lines.append('    for feat, thr, val, left, right in V_TREES:')
out_lines.append('        node = 0')
out_lines.append('        while feat[node] >= 0:')
out_lines.append('            node = left[node] if feats[feat[node]] < thr[node] else right[node]')
out_lines.append('        z += val[node]')
out_lines.append('    return 1.0 / (1.0 + _math.exp(-z))')

with open("value_4p_model.py", "w") as f:
    f.write("\n".join(out_lines))
print(f"Saved value_4p_model.py")

from google.colab import files
files.download("value_4p_model.py")
```

---

## 验证

下回来后：

```bash
# 把模型放到项目根
mv ~/Downloads/value_4p_model.py /Users/bor/Projects/orbit-wars/

# 验证 v42 加载它
.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
import value_4p_model
print('Loaded', len(value_4p_model.V_TREES), 'trees')
print('Sample predict:', value_4p_model.value_predict([0.5]*16))
"
```

应输出 trees 数 + 一个 0-1 的 P(win) 数。

然后 bench：

```bash
cp agents/v42_v4p.py main.py
.venv/bin/python scripts/tournament_4p.py --seeds 5 --opp lb1224 mlhybrid structured
```

---

## Troubleshooting

**"xgboost: no GPU detected"**：
- Colab 没切 GPU 运行时（Runtime → Change runtime type）
- 或 xgboost 装的是 CPU-only 版，重装：`!pip install xgboost --upgrade`

**OOM**：
- 数据太大时 GPU 内存爆。把 `tree_method` 改回 `"hist"` (CPU) + 删 `device: cuda`
- 或减 `max_depth`

**AUC 太低 (<0.7)**：
- 数据不够。回去多收几百局再训
- 或 features 不够 informative

**模型文件太大 (>10MB)**：
- Kaggle submission 有大小限制（agent < ? MB）
- 减 `--trees` 数到 500，或 `--depth` 到 5

---

## 整体时间预算

| 步骤 | 预计 |
|---|---|
| 本地数据收集（CPU 后台跑）| 5-8 小时 |
| 上传到 Colab | 1-2 分钟 |
| GPU 训练 800T × d7 | 2-5 分钟 |
| 下载 + 集成 | 5 分钟 |
| 本地 bench | 10-15 分钟 |
| **总计** | **6-9 小时**（其中 5-8h 是 CPU 后台，你可以睡觉）|

数据现在在跑（PID 78159）。等它跑完，再上 Colab 训练。
