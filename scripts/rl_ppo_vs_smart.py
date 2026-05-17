"""PPO training using smart_swarmer_v3 with randomized params as opponent."""
import argparse, json, math, os, random, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sim
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rl_candidate_gen as rcg
from rl_ppo import (PolicyValueNet, state_features, pad_candidates,
                     MAX_CANDS, CAND_DIM)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "opponents"))
# Import smart_swarmer_v3 as module — we'll vary its globals
import smart_swarmer_v3 as sw

import torch, torch.nn as nn, torch.nn.functional as F


def play_one_game(net, initial_state, num_players, device, ppo_player, max_steps=500):
    state = json.loads(json.dumps(initial_state))
    for k in ("seed", "num_players"): state.pop(k, None)
    state["initial_planets"] = state.get("initial_planets", state["planets"])
    state["comets"] = []; state["comet_planet_ids"] = []
    state["next_fleet_id"] = 1
    
    trajectory = []
    for step in range(max_steps):
        per_player = [None] * num_players
        for pl in range(num_players):
            if pl == ppo_player:
                feats = state_features(state, pl, num_players)
                cands = rcg.generate_candidates(state, pl, max_K=MAX_CANDS)
                my_pl = [p for p in state["planets"] if p[1] == pl]
                total_s = sum(p[5] for p in my_pl)
                cand_feats, mask = pad_candidates(cands, num_players, len(my_pl), total_s)
                ft = torch.tensor(feats, dtype=torch.float32, device=device).unsqueeze(0)
                cf = torch.tensor(cand_feats, dtype=torch.float32, device=device).unsqueeze(0)
                mk = torch.tensor(mask, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    logits, value = net(ft, cf, mk)
                probs = torch.sigmoid(logits[0])
                accept = torch.bernoulli(probs)
                log_p = torch.where(accept > 0,
                                     torch.log(probs.clamp(min=1e-8)),
                                     torch.log((1-probs).clamp(min=1e-8)))
                log_p = (log_p * mk[0]).sum()
                game_actions = []; used = {}
                for i, c in enumerate(cands):
                    if accept[i].item() > 0.5:
                        if used.get(c["mid"], 0) + c["ships"] > total_s: continue
                        game_actions.append([c["mid"], c["angle"], c["ships"]])
                        used[c["mid"]] = used.get(c["mid"], 0) + c["ships"]
                per_player[pl] = game_actions
                trajectory.append({"player": pl, "step": step, "state_feats": feats,
                                   "cand_feats": cand_feats, "mask": mask,
                                   "accept": accept.cpu().tolist(),
                                   "log_prob": log_p.item(), "value": value.item()})
            else:
                fake_obs = {"player": pl, "planets": state["planets"],
                            "step": step, "angular_velocity": state.get("angular_velocity", 0.04)}
                per_player[pl] = sw.agent(fake_obs)
        sim.step(state, per_player)
        owners = set(p[1] for p in state["planets"] if p[1] >= 0)
        if len(owners) <= 1: break
    
    counts = [0]*num_players
    for p in state["planets"]:
        if 0 <= p[1] < num_players: counts[p[1]] += 1
    winner = max(range(num_players), key=lambda i: counts[i])
    reward = 1.0 if winner == ppo_player else -1.0
    return trajectory, reward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--games_per_iter", type=int, default=16)
    ap.add_argument("--ppo_epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--init", default=None)
    ap.add_argument("--ckpt_dir", default="checkpoints_smart")
    args = ap.parse_args()
    
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)
    states_file = Path(__file__).resolve().parent.parent / "data" / "initial_states_4p.json"
    with open(states_file) as f: initial_states = json.load(f)
    print(f"Loaded {len(initial_states)} initial states")
    
    net = PolicyValueNet().to(args.device)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=args.device), strict=False)
        print(f"Warm-start from {args.init}")
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    for it in range(args.iters):
        t0 = time.time()
        all_traj = []
        wins = 0
        for g in range(args.games_per_iter):
            init = random.choice(initial_states)
            ppo_pos = g % 4
            # Randomize swarmer params per game
            sw.MAX_LAUNCHES = random.randint(8, 25)
            sw.BUF_FRAC = random.uniform(0.3, 1.0)
            traj, reward = play_one_game(net, init, 4, args.device, ppo_pos)
            last_v = 0.0; gae = 0.0
            for i in range(len(traj)-1, -1, -1):
                e = traj[i]; v = e["value"]
                r = reward if i == len(traj)-1 else 0.0
                delta = r + args.gamma * last_v - v
                gae = delta + args.gamma * args.lam * gae
                e["advantage"] = gae; e["return"] = gae + v
                last_v = v
            all_traj.extend(traj)
            wins += int(reward > 0)
        play_t = time.time() - t0
        
        if not all_traj: continue
        t1 = time.time()
        states = torch.tensor([e["state_feats"] for e in all_traj], dtype=torch.float32, device=args.device)
        cands = torch.tensor([e["cand_feats"] for e in all_traj], dtype=torch.float32, device=args.device)
        masks = torch.tensor([e["mask"] for e in all_traj], dtype=torch.float32, device=args.device)
        accepts = torch.tensor([e["accept"] for e in all_traj], dtype=torch.float32, device=args.device)
        old_lp = torch.tensor([e["log_prob"] for e in all_traj], dtype=torch.float32, device=args.device)
        advs = torch.tensor([e["advantage"] for e in all_traj], dtype=torch.float32, device=args.device)
        rets = torch.tensor([e["return"] for e in all_traj], dtype=torch.float32, device=args.device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        
        for _ in range(args.ppo_epochs):
            perm = torch.randperm(len(all_traj))
            for i in range(0, len(all_traj), args.batch):
                idx = perm[i:i+args.batch]
                new_logits, new_value = net(states[idx], cands[idx], masks[idx])
                new_probs = torch.sigmoid(new_logits)
                new_lp = torch.where(accepts[idx] > 0,
                                       torch.log(new_probs.clamp(min=1e-8)),
                                       torch.log((1-new_probs).clamp(min=1e-8)))
                new_lp = (new_lp * masks[idx]).sum(-1)
                ratio = torch.exp(new_lp - old_lp[idx])
                clipped = torch.clamp(ratio, 1-args.clip, 1+args.clip)
                pl_loss = -torch.min(ratio * advs[idx], clipped * advs[idx]).mean()
                vl = F.mse_loss(new_value, rets[idx])
                ent = -(new_probs * torch.log(new_probs.clamp(min=1e-8)) +
                          (1-new_probs) * torch.log((1-new_probs).clamp(min=1e-8)))
                ent = (ent * masks[idx]).sum(-1).mean()
                loss = pl_loss + 0.5*vl - 0.02*ent
                opt.zero_grad(); loss.backward(); opt.step()
        train_t = time.time() - t1
        wr = wins / args.games_per_iter
        print(f"[iter {it+1}/{args.iters}] WR_vs_smart={wr:.2f} "
              f"avg_steps={len(all_traj)/args.games_per_iter:.0f} "
              f"play={play_t:.1f}s train={train_t:.1f}s", flush=True)
        if (it+1) % 10 == 0:
            ck = f"{args.ckpt_dir}/ppo_smart_iter_{it+1}.pt"
            torch.save(net.state_dict(), ck)
            print(f"  ✅ saved {ck}", flush=True)


if __name__ == "__main__":
    main()
