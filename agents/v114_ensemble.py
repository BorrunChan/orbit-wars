"""v114: ENSEMBLE — route to v98 or v110 based on opp launch volume.

Hypothesis (from session iterations 1-6):
- v98 wins vs structured opps (47% on standard 4-opp pool)
- v110 (proto1000 fork) wins head-to-head vs v98 (6/6) but loses vs structured
- Rock-paper-scissors: no universal winner
- v114 = adaptive routing based on early-game opp launch_rate detection

Routing:
- Track max opp fleets in flight across early turns (proxy for launch_rate)
- After step 15, commit to one framework (sticky decision)
- High volume (max_fleets >= 6) -> v110 proto-style
- Otherwise -> v98 sim-greedy

Early-fire route: if very high volume detected by step 8 (max_fleets >= 5),
commit early to v110 (don't waste turns playing v98 vs proto-style).
"""
import os
import sys

# Make sibling modules importable (kaggle env adds the loaded file's dir to
# sys.path, so this is mainly for safety + clarity).
_AGENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

# Project root for shot_model / value_4p_model (at root, not agents/).
# v98_horz2p loads them via try/except - falls back if missing.
_PROJECT_ROOT = os.path.dirname(_AGENT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import v98_horz2p  # noqa: E402
import v110_proto_fork  # noqa: E402


_state = {
    "max_opp_fleets_seen": 0,
    "last_step": -1,
    "decision": None,
}

DECISION_STEP_DEFAULT = 15
DECISION_STEP_EARLY = 8
EARLY_FIRE_THRESHOLD = 5
LATE_DECISION_THRESHOLD = 6


def _reset_if_new_game(step):
    """Step typically increases monotonically within a game; if step regresses
    or jumps to 0/1, we're in a new game — reset state."""
    if step < _state["last_step"] or step <= 1:
        _state["max_opp_fleets_seen"] = 0
        _state["decision"] = None
    _state["last_step"] = step


def agent(obs):
    if isinstance(obs, dict):
        getf = obs.get
    else:
        getf = lambda k, default=None: getattr(obs, k, default)
    step = getf("step", 0) or 0
    raw_fleets = getf("fleets", []) or []
    player = getf("player", 0)

    _reset_if_new_game(step)

    opp_in_flight = sum(1 for f in raw_fleets
                         if f[1] != player and f[1] >= 0)
    if opp_in_flight > _state["max_opp_fleets_seen"]:
        _state["max_opp_fleets_seen"] = opp_in_flight

    if _state["decision"] is None:
        if step >= DECISION_STEP_EARLY and \
                _state["max_opp_fleets_seen"] >= EARLY_FIRE_THRESHOLD:
            _state["decision"] = "v110"
        elif step >= DECISION_STEP_DEFAULT:
            if _state["max_opp_fleets_seen"] >= LATE_DECISION_THRESHOLD:
                _state["decision"] = "v110"
            else:
                _state["decision"] = "v98"

    if _state["decision"] == "v110":
        return v110_proto_fork.agent(obs)
    return v98_horz2p.agent(obs)
