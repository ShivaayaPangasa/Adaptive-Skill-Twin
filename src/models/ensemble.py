import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "dataset"))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from marker_filter import RAW_MARKERS
from graph import build_adjacency
from lstm_model import LSTMClassifier
from stgcn_model import STGCN

INDEX_PATH = "data/processed_index.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_split(split, labels_order=None):
    with open(INDEX_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    labels = labels_order or sorted(set(r["label"] for r in csv.DictReader(open(INDEX_PATH, newline="", encoding="utf-8"))))
    label_to_idx = {l: i for i, l in enumerate(labels)}
    X = np.array([np.load(r["npy_path"]) for r in rows])
    y = np.array([label_to_idx[r["label"]] for r in rows])
    return X, y, labels


def rf_features(X):
    mean = X.mean(axis=1).reshape(len(X), -1)
    std = X.std(axis=1).reshape(len(X), -1)
    minv = X.min(axis=1).reshape(len(X), -1)
    maxv = X.max(axis=1).reshape(len(X), -1)
    path_len = np.linalg.norm(np.diff(X, axis=1), axis=3).sum(axis=1)
    return np.concatenate([mean, std, minv, maxv, path_len], axis=1)


if __name__ == "__main__":
    X_train, y_train, labels = load_split("train")
    X_test, y_test, _ = load_split("test", labels_order=labels)
    print(f"Test set: {len(X_test)} samples")

    # LSTM
    lstm = LSTMClassifier(input_size=117, num_classes=len(labels)).to(DEVICE)
    lstm.load_state_dict(torch.load("outputs/lstm_baseline.pt", map_location=DEVICE, weights_only=True))
    lstm.eval()

    # ST-GCN
    A = build_adjacency()
    stgcn = STGCN(A, in_channels=3, num_classes=len(labels)).to(DEVICE)
    stgcn.load_state_dict(torch.load("outputs/stgcn_baseline.pt", map_location=DEVICE, weights_only=True))
    stgcn.eval()

    # Random Forest (retrained here on train split for consistency)
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(rf_features(X_train), y_train)

    with torch.no_grad():
        x_flat = torch.from_numpy(X_test.reshape(len(X_test), 100, -1)).float().to(DEVICE)
        lstm_probs = F.softmax(lstm(x_flat), dim=1).cpu().numpy()

        x_graph = torch.from_numpy(X_test).float().to(DEVICE)
        stgcn_probs = F.softmax(stgcn(x_graph), dim=1).cpu().numpy()

    rf_probs = rf.predict_proba(rf_features(X_test))

    # Individual results
    for name, probs in [("LSTM", lstm_probs), ("ST-GCN", stgcn_probs), ("Random Forest", rf_probs)]:
        preds = probs.argmax(axis=1)
        print(f"{name}: acc={accuracy_score(y_test, preds):.3f}  macro_f1={f1_score(y_test, preds, average='macro'):.3f}")

    # Ensemble: simple averaged softmax
    ensemble_probs = (lstm_probs + stgcn_probs + rf_probs) / 3
    ensemble_preds = ensemble_probs.argmax(axis=1)
    print(f"\nENSEMBLE (avg of all 3): acc={accuracy_score(y_test, ensemble_preds):.3f}  "
          f"macro_f1={f1_score(y_test, ensemble_preds, average='macro'):.3f}")

    # Ensemble: ST-GCN + RF only (dropping the weakest model, LSTM)
    ensemble2_probs = (stgcn_probs + rf_probs) / 2
    ensemble2_preds = ensemble2_probs.argmax(axis=1)
    print(f"ENSEMBLE (ST-GCN + RF): acc={accuracy_score(y_test, ensemble2_preds):.3f}  "
          f"macro_f1={f1_score(y_test, ensemble2_preds, average='macro'):.3f}")