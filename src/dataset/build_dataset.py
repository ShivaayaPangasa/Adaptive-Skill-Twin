import csv
import numpy as np
from pathlib import Path
from preprocess import preprocess_file

MANIFEST_PATH = Path("data/manifest_split.csv")
OUT_DIR = Path("data/processed")
INDEX_PATH = Path("data/processed_index.csv")

if __name__ == "__main__":
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index_rows, n_failed = [], 0

    for i, row in enumerate(rows, start=1):
        out_path = OUT_DIR / f"{row['participant']}_{row['session']}_{row['exercise']}_{row['trial']}_{i}.npy"
        try:
            tensor = preprocess_file(Path(row["filepath"]))
            if not np.isfinite(tensor).all():
                raise ValueError("Non-finite values (NaN/Inf) in preprocessed tensor")
            np.save(out_path, tensor)
            index_rows.append({"npy_path": str(out_path), "participant": row["participant"],
                                "label": row["exercise"], "split": row["split"]})
        except Exception as e:
            n_failed += 1
            print(f"  failed on {row['filename']}: {e}")
        if i % 100 == 0:
            print(f"  processed {i}/{len(rows)}")

    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["npy_path", "participant", "label", "split"])
        writer.writeheader()
        writer.writerows(index_rows)

    print(f"\nDone. {len(index_rows)} succeeded, {n_failed} failed.")
    print(f"Index saved to {INDEX_PATH}")