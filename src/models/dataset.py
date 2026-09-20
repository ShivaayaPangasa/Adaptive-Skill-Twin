import csv
import numpy as np
import torch
from torch.utils.data import Dataset

INDEX_PATH = "data/processed_index.csv"


class KarateDataset(Dataset):
    def __init__(self, split, index_path=INDEX_PATH):
        with open(index_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.rows = [r for r in rows if r["split"] == split]
        self.labels = sorted(set(r["label"] for r in rows))
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        xyz = np.load(row["npy_path"])            # (100, 39, 3)
        x = xyz.reshape(xyz.shape[0], -1)          # (100, 117) - flatten markers*3 per frame
        y = self.label_to_idx[row["label"]]
        return torch.from_numpy(x).float(), y