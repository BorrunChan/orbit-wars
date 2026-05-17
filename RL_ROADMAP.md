# Orbit Wars RL Training Roadmap

Pipeline 已就绪。当前状态 + 后续步骤。

## ✅ 已就绪(本机连过来直接可用)

```
~/orbit_wars_rl/
├── sim.py                              # 模拟器(独立,无 kaggle_env 依赖)
├── data/
│   ├── initial_states_4p.json (4.1 MB) # 1000 局初始 state
│   └── rl_traj_v69.jsonl (~10K rows)   # v69 self-play trajectories
├── scripts/
│   ├── rl_policy.py                    # PolicyValueNet 定义(待用)
│   ├── rl_train.py                     # behavior cloning 训练
│   └── rl_selfplay_remote.py           # phase 1 ✅ value warm-up
└── value_net_v0.pt (77 KB)             # 已训:value net warm-up
```

## 🛠 已验证的工具

| 项 | 状态 |
|---|---|
| torch 2.11 + CUDA 13 | ✅ |
| 4× RTX 4090 (GPU 0/1 ~15GB 空闲) | ✅ |
| sim.step() | ✅(0.4ms/step,无 deps) |
| Self-play 随机 vs 随机 | ✅ 0.34s/局 |
| Value net 训练 | ✅ 30 epoch ~10s |
| 训练数据 IO | ✅ |

## 🔬 Phase 2:behavior cloning of v69(下一步)

输入:`data/rl_traj_v69.jsonl`(每行一个 turn 的 state + actions)
目标:网络学习 v69 在 state s 下会发几条船、发到哪。

```bash
cd ~/orbit_wars_rl
python3 scripts/rl_train.py --data data/rl_traj_v69.jsonl --epochs 50 --out policy_v1_bc.pt
```

时间预估:1-5 分钟。
预期:value loss 接近 v0 水平,policy 能近似 v69。

## 🚀 Phase 3:PPO self-play(主菜)

需要写的:`scripts/rl_ppo.py`。**这是大头,需要 200-500 行**。

### 训练循环

```python
for iteration in range(NUM_ITERS):
    # 1. Generate N_GAMES self-play games using current policy
    trajectories = []
    for game in range(N_GAMES):
        state = random.choice(initial_states)
        ep = []
        while not done:
            for player in range(4):
                feats = state_features(state, player, 4)
                # Generate candidates (port from main.py)
                cands = generate_candidates(state, player)
                # Policy: per-candidate accept logits
                logits, value = policy_net(feats, candidate_feats(cands))
                # Sample action set
                accept_probs = sigmoid(logits)
                accept_mask = bernoulli(accept_probs)
                actions = [cands[i] for i in range(len(cands)) if accept_mask[i]]
                ep.append((feats, candidate_feats, accept_mask, value))
            sim.step(state, [actions per player])
        # Compute returns + advantages
        winner = ...
        for entry in ep:
            entry.reward = +1 if winner else -1
        # GAE advantages
        ep = compute_advantages(ep, gamma=0.99, lam=0.95)
        trajectories.extend(ep)

    # 2. PPO update
    for ppo_epoch in range(N_PPO_EPOCHS):
        for batch in DataLoader(trajectories):
            new_logits, new_value = policy_net(batch.state, batch.cands)
            new_log_probs = log_prob(new_logits, batch.action)
            ratio = exp(new_log_probs - batch.old_log_probs)
            clipped = clip(ratio, 0.8, 1.2)
            policy_loss = -min(ratio * adv, clipped * adv).mean()
            value_loss = mse(new_value, batch.return)
            entropy = -prob * log_prob
            loss = policy_loss + 0.5*value_loss - 0.01*entropy
            loss.backward(); opt.step()

    # 3. Periodic eval vs v69-port
    if iteration % 10 == 0:
        wr = bench_policy_vs_v69_port(N=20)
        torch.save(policy_net, f"checkpoints/ppo_iter_{iteration}.pt")
```

### 关键难点

1. **Candidate generation 需要 port 一份到远端**(没 kaggle_environments,要用 sim 状态算)
2. **State + candidate 联合编码**:state 16d + per-candidate 9d
3. **Action 离散化**:每个候选独立 Bernoulli accept,combinatorial space 但 tractable
4. **Reward sparse**:用 value head + GAE 平滑
5. **训练时间预估**:1k iter × 100 games × 100 steps = 1000 万 sim.step ≈ **2-3 天单 GPU**

### 关键超参起步值

```
NUM_ITERS = 1000
N_GAMES = 32 (per iteration)
N_PPO_EPOCHS = 4
BATCH_SIZE = 256
LR = 3e-4
GAMMA = 0.99
LAM = 0.95
CLIP = 0.2
ENT_COEF = 0.01
```

## 📐 Phase 4: 导出 + 整合

PPO 模型完成后:
1. 导出网络为 pure-python(或保留 torch,看 Kaggle 限制)
2. 写 agent wrapper:`main_rl.py` 用 policy net 替换 greedy 决策
3. Local bench vs v69
4. 如果 > v69 → 替换 main.py

## ⚠️ 风险

- **Sparse reward + small state**:可能学不出有意义策略
- **Variance**:self-play 不稳定,可能 collapse
- **Candidate space variable**:每回合候选数不同,要处理 padding/masking
- **Wall-clock**:多日 GPU time,不一定有时间

## 🎯 推荐起步顺序

1. ✅ Phase 1 已完成(value warm-up)
2. **现在:Phase 2 behavior cloning**(快速,验证 pipeline)
3. **再:实现 candidate generation on remote**(port from main.py)
4. **然后:Phase 3 PPO loop**(主要工作)
5. **最后:导出 + bench**

每个 Phase 后存 checkpoint。
