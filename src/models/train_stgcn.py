import sys, csv
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / "dataset"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from graph import build_adjacency
from stgcn_model import STGCN

INDEX_PATH = "data/processed_index.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 100
BATCH_SIZE = 16
LR = 5e-4
WEIGHT_DECAY = 1e-4


def augment(xyz):
    angle = np.random.uniform(-15, 15) * np.pi / 180
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz = xyz @ R.T
    xyz = xyz + np.random.normal(0, 0.01, xyz.shape).astype(np.float32)
    return xyz


def add_velocity(xyz):
    vel = np.zeros_like(xyz)
    vel[1:] = xyz[1:] - xyz[:-1]
    return np.concatenate([xyz, vel], axis=-1)  # (T, V, 6)


class GraphKarateDataset(Dataset):
    def __init__(self, split, index_path=INDEX_PATH, augment_data=False):
        with open(index_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        self.labels = sorted(set(r["label"] for r in rows))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.augment_data = augment_data

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        xyz = np.load(row["npy_path"])
        if self.augment_data:
            xyz = augment(xyz)
        xyz = add_velocity(xyz)
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
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


if __name__ == "__main__":
    train_ds = GraphKarateDataset("train", augment_data=True)
    val_ds = GraphKarateDataset("val")
    test_ds = GraphKarateDataset("test")
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    counts = np.zeros(len(train_ds.labels))
    for r in train_ds.rows:
        counts[train_ds.label_to_idx[r["label"]]] += 1
    class_weights = torch.tensor(counts.sum() / (len(counts) * counts), dtype=torch.float32).to(DEVICE)
    print(f"Class counts: {counts}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    A = build_adjacency()
    model = STGCN(A, in_channels=6, num_classes=len(train_ds.labels)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1, best_state = -1, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_metrics = evaluate(model, val_loader)
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train_loss={total_loss/len(train_loader):.4f} "
                  f"| val_acc={val_metrics['accuracy']:.3f} | val_macro_f1={val_metrics['macro_f1']:.3f}")

    print(f"\nBest val macro_f1: {best_val_f1:.3f}")
    model.load_state_dict(best_state)

    print("\n=== FINAL TEST RESULTS (best checkpoint) ===")
    test_metrics = evaluate(model, test_loader)
    for k, v in test_metrics.items():
        print(f"{k}: {v}")

    torch.save(model.state_dict(), "outputs/stgcn_best.pt")
    print("\nModel saved to outputs/stgcn_best.pt")