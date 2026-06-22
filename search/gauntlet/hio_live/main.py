"""hio_live: hoard_io with its params ACTUALLY loaded.
hoard_io standalone ran code DEFAULT (params.json never read via getcwd). Here the
engine is an `import` submodule (has __file__), so it reads params.json → its
designed strategy (hoard屯兵 + proactive defense[death-code fixed] + econ rush +
orbital centrality) actually fires. See params-never-loaded-kaggle / kaggle-multifile-submission.
"""
import os
import sys

try:
    _d = os.path.dirname(os.path.abspath(__file__))
    if _d not in sys.path:
        sys.path.insert(0, _d)
except NameError:          # kaggle exec: agent dir already on sys.path
    pass

import hoard_engine as _E   # noqa: E402  (submodule has __file__ → reads params.json)


def agent(obs):
    return _E.agent(obs)
