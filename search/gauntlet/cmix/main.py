"""cmix (方案C): player-count-routed hybrid.
  2P -> lightver engine (single-size + 连续 catch-up + 终局抑制; 本地 2P 67% 最强)
  4P -> reyhan_1239 engine (4P multi-size + beta0 + 终局激进; Kaggle 真实 4P 41.7% 最强)
Each engine is loaded as an isolated module so their identically-named internals
(ProducerLiteConfig / plan_lite_waves / CONFIG_*) don't collide. They share one
orbit_lite (function sets verified identical). reyhan engine reads its own
params.json (1239-tuned) via _HERE.
"""
import os
import sys
import importlib.util

try:                       # local (cross_play / direct file load): add this file's dir
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path:
        sys.path.insert(0, _d)
except NameError:          # kaggle exec: kaggle_environments already appends the agent dir to sys.path
    pass

import lv_engine as _LV     # noqa: E402  (走 sys.path, 同 `import orbit_lite`)
import rey_engine as _REY   # noqa: E402

from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.planner_core import largest_initial_player_count

_PC = {"v": None}


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    obs_tensors = single_obs_to_tensor(obs, player_id=int(player))
    if _PC["v"] is None or bool((obs_tensors["step"] == 0).all()):
        _PC["v"] = int(largest_initial_player_count(obs_tensors))
    engine = _LV if _PC["v"] <= 2 else _REY
    return engine.agent(obs)
