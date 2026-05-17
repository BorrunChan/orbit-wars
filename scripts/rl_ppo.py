"""
PPO self-play training for Orbit Wars policy.

Architecture: Policy + Value net taking (state, candidates) → (accept logits, value)
Plays N self-play games per iteration, computes GAE, does PPO updates.

Run on remote GPU:
    cd ~/orbit_wars_rl
    python3 scripts/rl_ppo.py --iters 100 --games_per_iter 16 --device cuda
"""
import argparse, json, math, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_candidate_gen as rcg

import torch
import torch.nn as nn
import torch.nn.functional as F


STATE_DIM = 16
CAND_DIM = 9
MAX_CANDS = 15


def state_features(state, player, num_players):
    planets = state["planets"]; fleets = state["fleets"]; step = state["step"]
    op = [0]*4; os_ = [0]*4; opd = [0]*4
    neutral_p = 0; my_xy = []
    for p in planets:
        o = p[1]
        if o == -1: neutral_p += 1
        elif 0 <= o < 4:
            op[o] += 1; os_[o] += p[5]; opd[o] += p[6]
            if o == player: my_xy.append((p[2], p[3]))
    of = [0]*4
    for f in fleets:
        if 0 <= f[1] < 4: of[f[1]] += f[6]
    my_s = os_[player] + of[player]
    my_p = op[player]; my_pd = opd[player]; my_fs = of[player]
    if my_xy:
        cx = sum(x for x,_ in my_xy)/len(my_xy)
        cy = sum(y for _,y in my_xy)/len(my_xy)
        my_cent = math.hypot(cx-50, cy-50)
    else: my_cent = 0
    opp_i = [i for i in range(num_players) if i != player]
    opps_s = [os_[i]+of[i] for i in opp_i]
    opp_ts = sum(opps_s)
    max_os = max(opps_s, default=0)
    max_op = max([op[i] for i in opp_i], default=0)
    max_opd = max([opd[i] for i in opp_i], default=0)
    opp_tp = sum(op[i] for i in opp_i)
    opp_tpd = sum(opd[i] for i in opp_i)
    opp_fs = sum(of[i] for i in opp_i)
    opp_alive = sum(1 for i in opp_i if op[i]>0 or os_[i]>0)
    min_ed = 100.0
    for mx, my in my_xy:
        for p in planets:
            if p[1] != player and p[1] != -1:
                d = math.hypot(mx-p[2], my-p[3])
                if d < min_ed: min_ed = d
    tot_s = max(1, my_s+opp_ts); tot_p = max(1, my_p+opp_tp+neutral_p)
    tot_pd = max(1, my_pd+opp_tpd); tot_fs = max(1, my_fs+opp_fs)
    sd = lambda a,b: a/b if b>1e-9 else 0
    return [
        step/500, 1.0 if num_players==2 else 0, 1.0 if num_players==4 else 0,
        sd(my_s, tot_s), sd(my_p, tot_p), sd(my_pd, tot_pd),
        my_cent/60, sd(max_os, tot_s), sd(max_op, tot_p), sd(max_opd, tot_pd),
        opp_alive/max(1,num_players-1), min(1.0, min_ed/100),
        sd(my_s-opp_ts, tot_s), sd(my_p-opp_tp, tot_p),
        sd(my_pd-opp_tpd, tot_pd), sd(my_fs, tot_fs),
    ]


class PolicyValueNet(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.state_enc = nn.Sequential(
            nn.Linear(STATE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.cand_enc = nn.Sequential(
            nn.Linear(CAND_DIM, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(2*hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Tanh(),
        )

    def forward(self, state_feats, cand_feats, cand_mask):
        """
        state_feats: (B, 16)
        cand_feats:  (B, N, 9) — padded with zeros for unused slots
        cand_mask:   (B, N) — 1 for real candidates, 0 for pad
        Returns:
          logits: (B, N) — set to -inf at masked positions
          value:  (B,)
        """
        s = self.state_enc(state_feats)
        B, N, _ = cand_feats.shape
        c = self.cand_enc(cand_feats.reshape(B*N, -1)).reshape(B, N, -1)
        s_exp = s.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([s_exp, c], dim=-1)
        logits = self.policy_head(x).squeeze(-1)
        logits = logits.masked_fill(cand_mask == 0, -1e9)
        value = self.value_head(s).squeeze(-1)
        return logits, value


def pad_candidates(cands, num_players, my_planet_count, total_ships):
    """Pad to MAX_CANDS, return (feats, mask)."""
    feats = []
    for c in cands[:MAX_CANDS]:
        feats.append(rcg.candidate_features(c, my_planet_count, total_ships,
                                              num_players))
    while len(feats) < MAX_CANDS:
        feats.append([0.0]*CAND_DIM)
    mask = [1]*min(len(cands), MAX_CANDS) + [0]*max(0, MAX_CANDS - len(cands))
    return feats, mask


def play_one_game(net, initial_state, num_players, device, deterministic=False,
                   max_steps=500):
    """Run one full self-play game (all 4 players use same net). Return list of
    (state_feats, cand_feats, cand_mask, accept_mask, log_prob, value, player)
    snapshots, plus end reward per player."""
    state = json.loads(json.dumps(initial_state))
    for k in ("seed", "num_players"):
        state.pop(k, None)
    state["initial_planets"] = state.get("initial_planets", state["planets"])
    state["comets"] = []; state["comet_planet_ids"] = []
    state["next_fleet_id"] = 1

    trajectory = []  # per-(turn, player) snapshot
    for step in range(max_steps):
        # All players act simultaneously
        per_player_actions = []
        for pl in range(num_players):
            feats = state_features(state, pl, num_players)
            cands = rcg.generate_candidates(state, pl, max_K=MAX_CANDS)
            my_pl = [p for p in state["planets"] if p[1] == pl]
            my_pl_count = len(my_pl)
            total_ships = sum(p[5] for p in my_pl)
            cand_feats, mask = pad_candidates(cands, num_players,
                                                my_pl_count, total_ships)

            ft = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
            cf = torch.tensor(cand_feats, dtype=torch.float32, device=device).unsqueeze(0)
            mk = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, value = net(ft, cf, mk)
            probs = torch.sigmoid(logits[0])
            if deterministic:
                accept = (probs > 0.5).float()
            else:
                accept = torch.bernoulli(probs)
            # Compute log_prob: sum log p(a) over decisions
            log_p = torch.where(accept > 0,
                                 torch.log(probs.clamp(min=1e-8)),
                                 torch.log((1 - probs).clamp(min=1e-8)))
            log_p = (log_p * mk[0]).sum()

            # Translate to game actions
            game_actions = []
            used_planets = {}  # mid -> ships used
            for i, c in enumerate(cands):
                if accept[i].item() > 0.5:
                    if used_planets.get(c["mid"], 0) + c["ships"] > total_ships:
                        continue
                    game_actions.append([c["mid"], c["angle"], c["ships"]])
                    used_planets[c["mid"]] = used_planets.get(c["mid"], 0) + c["ships"]
            per_player_actions.append(game_actions)

            trajectory.append({
                "player": pl, "step": step,
                "state_feats": feats, "cand_feats": cand_feats, "mask": mask,
                "accept": accept.cpu().tolist(),
                "log_prob": log_p.item(),
                "value": value.item(),
            })

        sim.step(state, per_player_actions)
        # Termination
        owners = set(p[1] for p in state["planets"] if p[1] >= 0)
        if len(owners) <= 1:
            break

    # End-game reward: winner +1, others -1
    counts = [0]*num_players
    for p in state["planets"]:
        if 0 <= p[1] < num_players: counts[p[1]] += 1
    winner = max(range(num_players), key=lambda i: counts[i])
    rewards = [(1.0 if i == winner else -1.0) for i in range(num_players)]
    return trajectory, rewards


def compute_advantages(trajectory, rewards, num_players, gamma=0.99, lam=0.95):
    """Per-player GAE."""
    by_player = [[] for _ in range(num_players)]
    for entry in trajectory:
        by_player[entry["player"]].append(entry)
    advs = [0.0] * len(trajectory)
    rets = [0.0] * len(trajectory)
    for pl in range(num_players):
        ep = by_player[pl]
        if not ep: continue
        last_value = 0.0
        gae = 0.0
        # Reverse iterate
        for i in range(len(ep)-1, -1, -1):
            entry = ep[i]
            v = entry["value"]
            r = rewards[pl] if i == len(ep)-1 else 0.0
            delta = r + gamma * last_value - v
            gae = delta + gamma * lam * gae
            entry["advantage"] = gae
            entry["return"] = gae + v
            last_value = v
    return trajectory


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--games_per_iter", type=int, default=8)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", default=None, help="optional warm-start .pt")
    ap.add_argument("--ckpt_dir", default="checkpoints")
    args = ap.parse_args()

    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {args.device}")
    states_file = Path(__file__).resolve().parent.parent / "data" / "initial_states_4p.json"
    with open(states_file) as f:
        initial_states = json.load(f)
    print(f"Loaded {len(initial_states)} initial states")

    net = PolicyValueNet().to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device),
                             strict=False)
        print(f"Warm-started from {args.init}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    for it in range(args.iters):
        t0 = time.time()
        # 1. Self-play
        all_traj = []
        wins = 0; total = 0
        for g in range(args.games_per_iter):
            init = random.choice(initial_states)
            traj, rewards = play_one_game(net, init, num_players=4,
                                            device=args.device)
            traj = compute_advantages(traj, rewards, num_players=4,
                                       gamma=args.gamma, lam=args.lam)
            all_traj.extend(traj)
            wins += sum(1 for r in rewards if r > 0)
            total += 4
        play_t = time.time() - t0

        # 2. PPO update
        t1 = time.time()
        n = len(all_traj)
        states = torch.tensor([e["state_feats"] for e in all_traj],
                                dtype=torch.float32, device=args.device)
        cands = torch.tensor([e["cand_feats"] for e in all_traj],
                                dtype=torch.float32, device=args.device)
        masks = torch.tensor([e["mask"] for e in all_traj],
                                dtype=torch.float32, device=args.device)
        accepts = torch.tensor([e["accept"] for e in all_traj],
                                dtype=torch.float32, device=args.device)
        old_lp = torch.tensor([e["log_prob"] for e in all_traj],
                                dtype=torch.float32, device=args.device)
        advantages = torch.tensor([e["advantage"] for e in all_traj],
                                    dtype=torch.float32, device=args.device)
        returns = torch.tensor([e["return"] for e in all_traj],
                                dtype=torch.float32, device=args.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.ppo_epochs):
            perm = torch.randperm(n)
            for i in range(0, n, args.batch):
                idx = perm[i:i+args.batch]
                new_logits, new_value = net(states[idx], cands[idx], masks[idx])
                new_probs = torch.sigmoid(new_logits)
                new_lp = torch.where(accepts[idx] > 0,
                                       torch.log(new_probs.clamp(min=1e-8)),
                                       torch.log((1-new_probs).clamp(min=1e-8)))
                new_lp = (new_lp * masks[idx]).sum(-1)
                ratio = torch.exp(new_lp - old_lp[idx])
                clipped = torch.clamp(ratio, 1-args.clip, 1+args.clip)
                policy_loss = -torch.min(ratio * advantages[idx],
                                            clipped * advantages[idx]).mean()
                value_loss = F.mse_loss(new_value, returns[idx])
                # Entropy on Bernoulli per cand
                ent = -(new_probs * torch.log(new_probs.clamp(min=1e-8)) +
                          (1-new_probs) * torch.log((1-new_probs).clamp(min=1e-8)))
                ent = (ent * masks[idx]).sum(-1).mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * ent
                opt.zero_grad(); loss.backward(); opt.step()
        train_t = time.time() - t1

        wr = wins / total
        avg_steps = len(all_traj) / args.games_per_iter / 4
        print(f"[iter {it+1}/{args.iters}] WR={wr:.2f} avg_steps={avg_steps:.0f} "
              f"play={play_t:.1f}s train={train_t:.1f}s loss={loss.item():.3f} "
              f"v_loss={value_loss.item():.3f}",
              flush=True)
        if (it+1) % 10 == 0:
            ck = f"{args.ckpt_dir}/ppo_iter_{it+1}.pt"
            torch.save(net.state_dict(), ck)
            print(f"  ✅ saved {ck}", flush=True)


if __name__ == "__main__":
    main()
