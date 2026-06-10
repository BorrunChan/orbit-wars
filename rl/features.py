"""Shared state features for the learned value function.

Used BOTH in training (label states from self-play) and at inference (the
deployed value-agent). Pure-Python on the raw obs planet list so it is portable
into the Kaggle submission with no extra deps. Keep FEATURE_ORDER stable — the
trained weights depend on it.

Planet row layout (obs['planets'][i]): [id, owner, x, y, ?, ships, production].
owner: >=0 player id, <0 neutral. (Matches orbit_lite/obs.py columns.)
"""
from __future__ import annotations

FEATURE_ORDER = [
    "step_frac",          # game progress 0..1
    "my_pc_frac",         # my planets / total planets
    "my_prod_share",      # my production / all-owned production   (AUC 0.98 vs win)
    "my_ship_share",      # my ships / all-owned ships
    "pc_minus_leader",    # my planets - strongest opponent planets (scaled)
    "prod_minus_leader",  # my production - strongest opponent production (scaled)
    "my_ships_per_planet", # avg garrison (hoard signal)
    "neutral_share_left", # neutral planets / total (expansion room)
    "nP_is_4p",           # 1.0 if 4-player else 0.0
    "alive_opp_frac",     # alive opponents / (nP-1)
]
N_FEATURES = len(FEATURE_ORDER)


def extract_features(obs: dict, player_id: int, n_players: int, total_steps: int = 500) -> list[float]:
    """Return a fixed-length feature vector for ``player_id`` from one obs frame."""
    planets = obs.get("planets") or []
    step = float(obs.get("step", 0))

    my_pc = my_sh = my_pr = 0.0
    opp_pc = {}   # owner -> planet count
    opp_sh = opp_pr = 0.0
    neut_pc = 0.0
    total_pc = 0.0
    for x in planets:
        if len(x) < 7:
            continue
        total_pc += 1.0
        owner = x[1]; sh = x[5]; pr = x[6]
        if owner == player_id:
            my_pc += 1.0; my_sh += sh; my_pr += pr
        elif owner is not None and owner >= 0:
            opp_pc[owner] = opp_pc.get(owner, 0.0) + 1.0
            opp_sh += sh; opp_pr += pr
        else:
            neut_pc += 1.0

    lead_pc = max(opp_pc.values()) if opp_pc else 0.0
    # strongest-opponent production
    lead_pr = 0.0
    if opp_pc:
        pr_by_owner = {}
        for x in planets:
            if len(x) < 7 or x[1] is None or x[1] < 0 or x[1] == player_id:
                continue
            pr_by_owner[x[1]] = pr_by_owner.get(x[1], 0.0) + x[6]
        lead_pr = max(pr_by_owner.values()) if pr_by_owner else 0.0

    tot_pr = my_pr + opp_pr + 1e-6
    tot_sh = my_sh + opp_sh + 1e-6
    tot_pc = total_pc + 1e-6
    n_alive_opp = float(len(opp_pc))

    return [
        step / float(total_steps),
        my_pc / tot_pc,
        my_pr / tot_pr,
        my_sh / tot_sh,
        (my_pc - lead_pc) / 10.0,
        (my_pr - lead_pr) / 20.0,
        (my_sh / (my_pc + 1e-6)) / 100.0,
        neut_pc / tot_pc,
        1.0 if n_players >= 4 else 0.0,
        n_alive_opp / max(1.0, n_players - 1.0),
    ]
