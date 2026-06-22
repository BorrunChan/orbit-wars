"""lz_route (分治路由版): player-count-routed comet 连招.
  2P -> p2_engine (agent_main base + beta3.0; 本地最强 2P, lightver 83 攻克)
  4P -> p4_engine (hoard producer base + 彗星价值评估 + 三计 + 清仓; 彗星连招)
两 engine 各自独立模块(同名内部 ProducerLiteConfig/plan_lite_waves/CONFIG_* 不冲突),
共享一个 orbit_lite(已验证 a_comet/hoard_cval orbit_lite 完全相同)。默认全硬编码,
不依赖 params.json(规避 params-never-loaded bug)。
"""
import os
import sys
import importlib.util
try:                       # local file load: add this file's dir to sys.path
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path:
        sys.path.insert(0, _d)
except NameError:          # kaggle exec: kaggle_environments already appends agent dir
    pass
import p2_engine as _P2     # noqa: E402  (走 sys.path, 同 `import orbit_lite`)
import p4_engine as _P4     # noqa: E402
from orbit_lite.adapter import single_obs_to_tensor
from orbit_lite.planner_core import largest_initial_player_count

_PC = {"v": None}


def agent(obs):
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    obs_tensors = single_obs_to_tensor(obs, player_id=int(player))
    if _PC["v"] is None or bool((obs_tensors["step"] == 0).all()):
        _PC["v"] = int(largest_initial_player_count(obs_tensors))
    engine = _P2 if _PC["v"] <= 2 else _P4
    return engine.agent(obs)
