import os, sys
try:
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path:
        sys.path.insert(0, _d)
except NameError:
    pass
import hoard_engine as _E   # noqa: E402  (子模块有 __file__ -> 读 params.json, hoard 生效)
def agent(obs):
    return _E.agent(obs)
