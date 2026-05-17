"""
PPO-policy-driven agent. Loads checkpoints/ppo_iter_*.pt and uses policy to
select launches. Same candidate generation as v69.
"""
import math, os, sys
from pathlib import Path

# Find ROOT — when loaded by kaggle_environments, __file__ is in agents/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import sim as _sim
from kaggle_environments.envs.orbit_wars.orbit_wars import (
    Planet, Fleet, CENTER, ROTATION_RADIUS_LIMIT, BOARD_SIZE, SUN_RADIUS,
)

# Try to load torch + ppo net
_NET = None
try:
    import torch
    import torch.nn as nn
    # Inline a copy of PolicyValueNet (must match rl_ppo.py)
    STATE_DIM = 16
    CAND_DIM = 9
    MAX_CANDS = 15

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
            s = self.state_enc(state_feats)
            B, N, _ = cand_feats.shape
            c = self.cand_enc(cand_feats.reshape(B*N, -1)).reshape(B, N, -1)
            s_exp = s.unsqueeze(1).expand(-1, N, -1)
            x = torch.cat([s_exp, c], dim=-1)
            logits = self.policy_head(x).squeeze(-1)
            logits = logits.masked_fill(cand_mask == 0, -1e9)
            value = self.value_head(s).squeeze(-1)
            return logits, value

    _NET = PolicyValueNet()
    ckpt = "/Users/bor/Projects/orbit-wars/checkpoints/ppo_smart_iter_10.pt"
    if os.path.exists(ckpt):
        _NET.load_state_dict(torch.load(ckpt, map_location="cpu"))
        _NET.eval()
    else:
        _NET = None
except ImportError:
    pass


# Import candidate generation logic from scripts/rl_candidate_gen.py
# But that uses sim module. For agent use we need slightly different state shape.
# Just port the core logic inline using kaggle obs format.

import rl_candidate_gen as rcg


def _state_features(obs, player, num_players):
    """16-d features."""
    planets = list(obs.planets or [])
    fleets = list(obs.fleets or [])
    step = obs.step
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
    opp_ts = sum(opps_s); opp_tp = sum(op[i] for i in opp_i)
    opp_tpd = sum(opd[i] for i in opp_i); opp_fs = sum(of[i] for i in opp_i)
    max_os = max(opps_s, default=0)
    max_op = max([op[i] for i in opp_i], default=0)
    max_opd = max([opd[i] for i in opp_i], default=0)
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


def agent(obs):
    if isinstance(obs, dict):
        getf = obs.get
    else:
        getf = lambda k, d=None: getattr(obs, k, d)
    player = getf("player", 0)
    raw_planets = getf("planets", []) or []
    raw_fleets = getf("fleets", []) or []
    all_owners = {p[1] for p in raw_planets if p[1] >= 0}
    for f in raw_fleets:
        if f[1] >= 0: all_owners.add(f[1])
    all_owners.add(player)
    max_p = max(all_owners) if all_owners else player
    num_players = 4 if max_p >= 2 else 2

    # If no policy net loaded, fallback to empty action
    if _NET is None:
        return []

    # Build sim-state dict for rcg
    state = {
        "planets": [list(p) for p in raw_planets],
        "fleets": [list(f) for f in raw_fleets],
        "step": getf("step", 0),
        "angular_velocity": getf("angular_velocity", 0.04) or 0.04,
        "initial_planets": [list(p) for p in (getf("initial_planets", []) or [])],
        "comets": [], "comet_planet_ids": [],
    }
    cands = rcg.generate_candidates(state, player, max_K=MAX_CANDS)
    if not cands:
        return []
    my_pl = [p for p in raw_planets if p[1] == player]
    total_s = sum(p[5] for p in my_pl)

    # Pad
    cand_feats = []
    for c in cands[:MAX_CANDS]:
        cand_feats.append(rcg.candidate_features(c, len(my_pl), total_s, num_players))
    while len(cand_feats) < MAX_CANDS:
        cand_feats.append([0.0]*CAND_DIM)
    mask = [1]*min(len(cands), MAX_CANDS) + [0]*max(0, MAX_CANDS - len(cands))

    feats = _state_features(obs, player, num_players)
    ft = torch.tensor(feats, dtype=torch.float32).unsqueeze(0)
    cf = torch.tensor(cand_feats, dtype=torch.float32).unsqueeze(0)
    mk = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits, _ = _NET(ft, cf, mk)
    probs = torch.sigmoid(logits[0])
    accept = (probs > 0.5).float()

    actions = []
    used = {}
    for i, c in enumerate(cands):
        if accept[i].item() > 0.5:
            if used.get(c["mid"], 0) + c["ships"] > total_s:
                continue
            actions.append([c["mid"], c["angle"], int(c["ships"])])
            used[c["mid"]] = used.get(c["mid"], 0) + c["ships"]
    return actions
