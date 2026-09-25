import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "dataset"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from marker_filter import RAW_MARKERS
from graph import build_adjacency
from lstm_model import LSTMClassifier
from stgcn_model import STGCN

INDEX_PATH = "data/processed_index.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NOISE_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.5]
SEED = 42


def load_test_data():
    with open(INDEX_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    labels = sorted(set(r["label"] for r in csv.DictReader(open(INDEX_PATH, newline="", encoding="utf-8"))))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    X = np.array([np.load(r["npy_path"]) for r in rows])  # (N, 100, 39, 3)
    y = np.array([label_to_idx[r["label"]] for r in rows])
    return X, y, labels


def occlude(X, frac, rng):
    """Zero out a random subset of markers (consistently across all frames
    of each sample), simulating dropped/occluded markers."""
    X = X.copy()
    n_markers = X.shape[2]
    n_drop = int(n_markers * frac)
    for i in range(X.shape[0]):
        drop_idx = rng.choice(n_markers, size=n_drop, replace=False)
        X[i, :, drop_idx, :] = 0.0
    return X


@torch.no_grad()
def eval_lstm(model, X, y):
    model.eval()
    x_flat = torch.from_numpy(X.reshape(X.shape[0], X.shape[1], -1)).float().to(DEVICE)
    preds = model(x_flat).argmax(dim=1).cpu().numpy()
    return accuracy_score(y, preds)


@torch.no_grad()
def eval_stgcn(model, X, y):
    model.eval()
    x = torch.from_numpy(X).float().to(DEVICE)
    preds = model(x).argmax(dim=1).cpu().numpy()
    return accuracy_score(y, preds)


if __name__ == "__main__":
    X, y, labels = load_test_data()
    print(f"Test set: {len(X)} samples, classes: {labels}")

    lstm = LSTMClassifier(input_size=117, num_classes=len(labels)).to(DEVICE)
    
    lstm.load_state_dict(
    torch.load(
        "outputs/lstm_best.pt",
        map_location=DEVICE
    )
)

    A = build_adjacency()
    stgcn = STGCN(A, in_channels=3, num_classes=len(labels)).to(DEVICE)
    
    stgcn.load_state_dict(
    torch.load(
        "outputs/stgcn_best.pt",
        map_location=DEVICE))

    rng = np.random.default_rng(SEED)
    print(f"\n{'Occlusion':>10} | {'LSTM acc':>10} | {'ST-GCN acc':>10}")
    print("-" * 36)
    for frac in NOISE_LEVELS:
        X_occ = occlude(X, frac, rng)
        lstm_acc = eval_lstm(lstm, X_occ, y)
        stgcn_acc = eval_stgcn(stgcn, X_occ, y)
        print(f"{frac*100:>9.0f}% | {lstm_acc:>10.3f} | {stgcn_acc:>10.3f}")