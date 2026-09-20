import csv
import numpy as np
import torch
from torch.utils.data import Dataset

INDEX_PATH = "data/processed_index.csv"


def augment(xyz):
    """Random small rotation (about the vertical axis) + Gaussian jitter.
    Applied only to training samples."""
    angle = np.random.uniform(-15, 15) * np.pi / 180
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
    xyz = xyz @ R.T
    xyz = xyz + np.random.normal(0, 0.01, xyz.shape).astype(np.float32)
    return xyz


def add_velocity(xyz):
    """Append frame-to-frame displacement as extra channels."""
    vel = np.zeros_like(xyz)
    vel[1:] = xyz[1:] - xyz[:-1]
    return np.concatenate([xyz, vel], axis=-1)  # (T, V, 6)


class KarateDataset(Dataset):
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
        x = xyz.reshape(xyz.shape[0], -1)  # (100, 234)
        y = self.label_to_idx[row["label"]]
        return torch.from_numpy(x).float(), y