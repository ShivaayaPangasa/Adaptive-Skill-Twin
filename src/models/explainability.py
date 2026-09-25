import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "dataset"))

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from marker_filter import RAW_MARKERS
from graph import build_adjacency
from stgcn_model import STGCN

INDEX_PATH = "data/processed_index.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Anatomical groupings for interpretable occlusion
GROUPS = {
    "head":        ["LFHD", "RFHD", "LBHD", "RBHD"],
    "torso":       ["C7", "T10", "CLAV", "STRN", "RBAK"],
    "left_arm":    ["LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN"],
    "right_arm":   ["RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN"],
    "pelvis":      ["LASI", "RASI", "LPSI", "RPSI"],
    "left_leg":    ["LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE"],
    "right_leg":   ["RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE"],
}


def load_test_data():
    with open(INDEX_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    labels = sorted(set(r["label"] for r in csv.DictReader(open(INDEX_PATH, newline="", encoding="utf-8"))))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    X = np.array([np.load(r["npy_path"]) for r in rows])
    y = np.array([label_to_idx[r["label"]] for r in rows])
    return X, y, labels


def occlude_group(X, marker_names):
    X = X.copy()
    idx = [RAW_MARKERS.index(m) for m in marker_names]
    X[:, :, idx, :] = 0.0
    return X


@torch.no_grad()
def eval_stgcn(model, X, y):
    model.eval()
    x = torch.from_numpy(X).float().to(DEVICE)
    preds = model(x).argmax(dim=1).cpu().numpy()
    return accuracy_score(y, preds)


if __name__ == "__main__":
    X, y, labels = load_test_data()

    A = build_adjacency()
    model = STGCN(A, in_channels=3, num_classes=len(labels)).to(DEVICE)
    
    model.load_state_dict(
    torch.load(
        "outputs/stgcn_best.pt",
        map_location=DEVICE))
    
    baseline_acc = eval_stgcn(model, X, y)
    print(f"Baseline (no occlusion) accuracy: {baseline_acc:.3f}\n")

    print(f"{'Group occluded':>15} | {'Accuracy':>10} | {'Drop':>8}")
    print("-" * 40)
    results = []
    for group_name, markers in GROUPS.items():
        X_occ = occlude_group(X, markers)
        acc = eval_stgcn(model, X_occ, y)
        drop = baseline_acc - acc
        results.append((group_name, acc, drop))
        print(f"{group_name:>15} | {acc:>10.3f} | {drop:>8.3f}")

    print("\nRanked by importance (largest accuracy drop = most important):")
    for name, acc, drop in sorted(results, key=lambda r: -r[2]):
        print(f"  {name}: -{drop:.3f}")