"""Small MLP value head: features -> P(win).

Train with torch (CUDA on the 3060). Deploy via a pure-Python forward pass from
exported JSON weights (no torch-load risk on the 1s/turn Kaggle path; torch is
available there but a tiny hand-rolled forward is faster and dependency-free).
"""
from __future__ import annotations
import json, math

try:
    import torch
    import torch.nn as nn

    class ValueNet(nn.Module):
        def __init__(self, n_in: int, hidden: int = 64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, hidden), nn.ReLU(),
                nn.Linear(hidden, hidden), nn.ReLU(),
                nn.Linear(hidden, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)  # logit

    def export_json(model: "ValueNet", path: str) -> None:
        w = {}
        for name, p in model.state_dict().items():
            w[name] = p.detach().cpu().tolist()
        json.dump(w, open(path, "w"))
except Exception:  # torch absent (e.g. partial env) — training-only import guard
    ValueNet = None  # type: ignore


# ---- pure-python deploy forward (no torch) --------------------------------
def _relu(v):
    return [x if x > 0 else 0.0 for x in v]


def _linear(v, W, b):
    # W: [out, in] (torch Linear weight), b: [out]
    out = []
    for o in range(len(W)):
        s = b[o]
        row = W[o]
        for i in range(len(v)):
            s += row[i] * v[i]
        out.append(s)
    return out


def load_weights(path: str) -> dict:
    return json.load(open(path))


def forward_prob(weights: dict, feats: list[float]) -> float:
    """P(win) from exported JSON weights. Mirrors ValueNet(Sequential 3-layer)."""
    v = feats
    v = _relu(_linear(v, weights["net.0.weight"], weights["net.0.bias"]))
    v = _relu(_linear(v, weights["net.2.weight"], weights["net.2.bias"]))
    v = _linear(v, weights["net.4.weight"], weights["net.4.bias"])
    logit = v[0]
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit))))
