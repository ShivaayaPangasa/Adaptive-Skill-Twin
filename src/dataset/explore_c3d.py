import ezc3d
from pathlib import Path

DATA_DIR = Path("data/raw")

def find_c3d_files(data_dir):
    """Recursively find every .c3d file under data_dir, regardless of folder structure."""
    return sorted(data_dir.rglob("*.c3d"))

def explore_c3d(path):
    c3d = ezc3d.c3d(str(path))

    marker_labels = c3d['parameters']['POINT']['LABELS']['value']
    sampling_rate = c3d['parameters']['POINT']['RATE']['value'][0]
    points = c3d['data']['points']

    n_markers = points.shape[1]
    n_frames = points.shape[2]

    print("=== C3D FILE SUMMARY ===")
    print(f"File: {path}")
    print(f"Number of markers: {n_markers}")
    print(f"Sampling rate: {sampling_rate} Hz")
    print(f"Number of frames: {n_frames}")
    print(f"Duration: {n_frames / sampling_rate:.2f} seconds")
    print(f"First 10 marker labels: {marker_labels[:10]}")

if __name__ == "__main__":
    files = find_c3d_files(DATA_DIR)

    if not files:
        print(f"No .c3d files found under {DATA_DIR}. Did you extract the zip into data/raw?")
    else:
        print(f"Found {len(files)} .c3d file(s):")
        for f in files:
            print(f"  {f}")
        print()

        # Explore the first one found
        explore_c3d(files[0])