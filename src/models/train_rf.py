import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

INDEX_PATH = "data/processed_index.csv"


def extract_features(xyz):
    """Per-marker, per-axis summary statistics: mean, std, min, max, and
    total path length (sum of frame-to-frame movement). Compact, standard
    hand-crafted representation for small-data classical ML baselines."""
    mean = xyz.mean(axis=0).flatten()
    std = xyz.std(axis=0).flatten()
    minv = xyz.min(axis=0).flatten()
    maxv = xyz.max(axis=0).flatten()
    path_len = np.linalg.norm(np.diff(xyz, axis=0), axis=2).sum(axis=0)
    return np.concatenate([mean, std, minv, maxv, path_len])


def load_split(split):
    with open(INDEX_PATH, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    X = np.array([extract_features(np.load(r["npy_path"])) for r in rows])
    # Cast labels to int so scikit-learn class_weight="balanced" works correctly
    y = np.array([int(r["label"]) for r in rows])
    return X, y


if __name__ == "__main__":
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")
    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Feature dim: {X_train.shape[1]}")

    # combine train+val for final fit since RF doesn't need a val loop the same way,
    # but we still report val separately first for transparency
    clf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    for name, X, y in [("VAL", X_val, y_val), ("TEST", X_test, y_test)]:
        preds = clf.predict(X)
        print(f"\n=== {name} RESULTS ===")
        print(f"accuracy: {accuracy_score(y, preds)}")
        print(f"macro_f1: {f1_score(y, preds, average='macro')}")
        print(f"precision: {precision_score(y, preds, average='macro', zero_division=0)}")
        print(f"recall: {recall_score(y, preds, average='macro', zero_division=0)}")
        print(f"confusion_matrix: {confusion_matrix(y, preds).tolist()}")