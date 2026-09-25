import csv
import numpy as np
import torch
from torch.utils.data import Dataset

INDEX_PATH = "data/processed_index.csv"


def augment(xyz):
    """Apply a small random rotation and Gaussian noise."""

    angle = np.random.uniform(-15, 15) * np.pi / 180

    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ], dtype=np.float32)

    xyz = xyz @ R.T

    noise = np.random.normal(
        0,
        0.01,
        xyz.shape
    ).astype(np.float32)

    xyz = xyz + noise

    return xyz


class KarateDataset(Dataset):

    def __init__(
        self,
        split,
        index_path=INDEX_PATH,
        augment_data=False
    ):

        with open(
            index_path,
            newline="",
            encoding="utf-8"
        ) as f:
            rows = list(csv.DictReader(f))

        self.rows = [
            r for r in rows
            if r["split"] == split
        ]

        self.labels = sorted(
            set(r["label"] for r in rows)
        )

        self.label_to_idx = {
            label: i
            for i, label in enumerate(self.labels)
        }

        self.augment_data = augment_data

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):

        row = self.rows[i]

        xyz = np.load(row["npy_path"])

        # Expected shape:
        # (100, 39, 3)

        if self.augment_data:
            xyz = augment(xyz)

        y = self.label_to_idx[row["label"]]

        return (
            torch.from_numpy(xyz).float(),
            y
        )