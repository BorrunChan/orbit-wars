"""cmix_fch: player-count-routed hybrid.
  2P -> lightver engine (single-size + catch-up + 终局抑制; 本地 2P 最强)
  4P -> f_ch engine (agent_main: multi-size + 真出逃 + 围魏 + 趁火打劫1.5 + 保守;
                     本地 4P 夺冠 41.2% 最强)
Each engine loaded as isolated module (identical internals don't collide); shared
orbit_lite. Multi-file load via `import` (走 sys.path) — see kaggle-multifile-submission.
"""
import os
import sys

try:                       # local (cross_play / direct file load): add this file's dir
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path:
        sys.path.insert(0, _d)
except NameError:          # kaggle exec: kaggle_environments already appends the agent dir to sys.path
    pass

import lv_engine as _LV      # noqa: E402  (2P)
import fch_engine as _F4     # noqa: E402  (4P, = agent_main / f_ch)

from orbit_lite.adapter import single_obs_to_tensor      # noqa: E402
from orbit_lite.planner_core import largest_initial_player_count  # noqa: E402

_PC = {"v": None}


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    obs_tensors = single_obs_to_tensor(obs, player_id=int(player))
    if _PC["v"] is None or bool((obs_tensors["step"] == 0).all()):
        _PC["v"] = int(largest_initial_player_count(obs_tensors))
    engine = _LV if _PC["v"] <= 2 else _F4
    return engine.agent(obs)
