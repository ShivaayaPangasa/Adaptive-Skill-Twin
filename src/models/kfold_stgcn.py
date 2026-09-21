import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "dataset"))

import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from graph import build_adjacency
from stgcn_model import STGCN

INDEX_PATH = "data/processed_index.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
K_FOLDS = 5
EPOCHS = 30
BATCH_SIZE = 16
LR = 5e-4
SEED = 42


class SimpleDataset(Dataset):
    def __init__(self, rows, label_to_idx):
        self.rows = rows
        self.label_to_idx = label_to_idx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        xyz = np.load(row["npy_path"])  # (100, 39, 3), no augmentation, no velocity
        y = self.label_to_idx[row["label"]]
        return torch.from_numpy(xyz).float(), y


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            preds = model(x).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
    return accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average="macro")


def train_one_fold(train_rows, val_rows, test_rows, labels, label_to_idx):
    train_loader = DataLoader(SimpleDataset(train_rows, label_to_idx), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SimpleDataset(val_rows, label_to_idx), batch_size=BATCH_SIZE)
    test_loader = DataLoader(SimpleDataset(test_rows, label_to_idx), batch_size=BATCH_SIZE)

    A = build_adjacency()
    model = STGCN(A, in_channels=3, num_classes=len(labels)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_val_f1, best_state = -1, None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        val_acc, val_f1 = evaluate(model, val_loader)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_acc, test_f1 = evaluate(model, test_loader)
    return test_acc, test_f1


if __name__ == "__main__":
    with open(INDEX_PATH, newline="", encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))

    labels = sorted(set(r["label"] for r in all_rows))
    label_to_idx = {l: i for i, l in enumerate(labels)}

    participants = sorted(set(r["participant"] for r in all_rows))
    rng = random.Random(SEED)
    rng.shuffle(participants)

    fold_size = len(participants) // K_FOLDS
    folds = [participants[i * fold_size:(i + 1) * fold_size] for i in range(K_FOLDS)]
    # put any remainder participants into the last fold
    remainder = participants[K_FOLDS * fold_size:]
    folds[-1].extend(remainder)

    accs, f1s = [], []

    for fold_i in range(K_FOLDS):
        test_p = set(folds[fold_i])
        remaining_p = [p for p in participants if p not in test_p]
        rng2 = random.Random(SEED + fold_i)
        rng2.shuffle(remaining_p)
        n_val = max(1, int(len(remaining_p) * 0.15))
        val_p = set(remaining_p[:n_val])
        train_p = set(remaining_p[n_val:])

        train_rows = [r for r in all_rows if r["participant"] in train_p]
        val_rows = [r for r in all_rows if r["participant"] in val_p]
        test_rows = [r for r in all_rows if r["participant"] in test_p]

        print(f"\nFold {fold_i+1}/{K_FOLDS} | train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} "
              f"(test participants: {sorted(test_p)})")

        acc, f1 = train_one_fold(train_rows, val_rows, test_rows, labels, label_to_idx)
        print(f"  -> test_acc={acc:.3f}  test_macro_f1={f1:.3f}")
        accs.append(acc)
        f1s.append(f1)

    print("\n=== K-FOLD SUMMARY (ST-GCN, participant-disjoint) ===")
    print(f"Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}  (per-fold: {[round(a,3) for a in accs]})")
    print(f"Macro-F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}  (per-fold: {[round(f,3) for f in f1s]})")