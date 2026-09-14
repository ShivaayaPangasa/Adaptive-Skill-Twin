import ezc3d
from pathlib import Path

DATA_DIR = Path("data/raw")

def find_c3d_files(data_dir):
    return sorted(data_dir.rglob("*.c3d"))

def inspect_all_markers(path):
    c3d = ezc3d.c3d(str(path))
    labels = c3d['parameters']['POINT']['LABELS']['value']

    print(f"File: {path}")
    print(f"Total points/markers: {len(labels)}\n")

    for i, label in enumerate(labels):
        print(f"{i:3d}: {label}")

if __name__ == "__main__":
    files = find_c3d_files(DATA_DIR)
    if files:
        inspect_all_markers(files[0])
    else:
        print("No files found.")