"""Train the value MLP on collected (features, win) data. CUDA if available.

Usage (3060 box):
    python rl/train_value.py --data rl/data/iter0.jsonl --out rl/data/value_iter0.json
Outputs JSON weights (for pure-Python deploy forward) + prints holdout AUC.
Gate: holdout AUC should be high (>0.9 expected) AND, more importantly, later
iterations' agents must beat earlier ones on Kaggle — AUC alone is not the goal.
"""
from __future__ import annotations
import argparse, json, random, sys, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
from rl.value_net import ValueNet, export_json
from rl.features import N_FEATURES


def load(path):
    X, y = [], []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        X.append(r["f"]); y.append(float(r["won"]))
    return X, y


def auc(model, X, y, device):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32, device=device))).cpu().tolist()
    pos = [pi for pi, yi in zip(p, y) if yi > 0.5]
    neg = [pi for pi, yi in zip(p, y) if yi <= 0.5]
    if not pos or not neg:
        return float("nan")
    rng = random.Random(0); c = t = 0
    for _ in range(30000):
        a = rng.choice(pos); b = rng.choice(neg)
        c += 1 if a > b else (0.5 if a == b else 0); t += 1
    return c / t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="one or more jsonl files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    X, y = [], []
    for d in args.data:
        Xi, yi = load(d); X += Xi; y += yi
    assert X and len(X[0]) == N_FEATURES, f"feature dim mismatch: {len(X[0]) if X else 0} vs {N_FEATURES}"
    idx = list(range(len(X))); random.Random(0).shuffle(idx)
    cut = int(0.85 * len(idx))
    tr, ho = idx[:cut], idx[cut:]
    Xtr = torch.tensor([X[i] for i in tr], dtype=torch.float32, device=device)
    ytr = torch.tensor([y[i] for i in tr], dtype=torch.float32, device=device)
    Xho = [X[i] for i in ho]; yho = [y[i] for i in ho]

    model = ValueNet(N_FEATURES, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = Xtr.shape[0]
    for ep in range(args.epochs):
        model.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, args.bs):
            b = perm[s:s + args.bs]
            opt.zero_grad()
            loss = lossf(model(Xtr[b]), ytr[b]); loss.backward(); opt.step()
    a = auc(model, Xho, yho, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    export_json(model, args.out)
    print(f"device={device} samples={len(X)} holdout_AUC={a:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
