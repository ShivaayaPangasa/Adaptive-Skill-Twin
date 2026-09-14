import ezc3d
from pathlib import Path

DATA_DIR = Path("data/raw")

# The 39 raw reflective markers from the Vicon Plug-in Gait full-body set.
# Everything after these in the file is a *computed/modeled* output (joint
# angles, forces, moments, centers of mass) - not something a phone camera
# could ever capture, so we exclude it from the modeling pipeline.
RAW_MARKERS = [
    "LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN", "RBAK",
    "LSHO", "LUPA", "LELB", "LFRM", "LWRA", "LWRB", "LFIN",
    "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN",
    "LASI", "RASI", "LPSI", "RPSI",
    "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
    "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE",
]

def find_c3d_files(data_dir):
    return sorted(data_dir.rglob("*.c3d"))

def load_raw_markers(path):
    """Load a C3D file and return only the 39 raw marker trajectories,
    discarding Plug-in Gait's ~179 modeled/computed points."""
    c3d = ezc3d.c3d(str(path))
    all_labels = c3d['parameters']['POINT']['LABELS']['value']
    points = c3d['data']['points']  # shape: (4, n_all_points, n_frames)

    found_markers = [m for m in RAW_MARKERS if m in all_labels]
    missing = [m for m in RAW_MARKERS if m not in all_labels]
    indices = [all_labels.index(m) for m in found_markers]

    raw_points = points[:, indices, :]  # (4, 39, n_frames)

    return raw_points, found_markers, missing

if __name__ == "__main__":
    files = find_c3d_files(DATA_DIR)
    if not files:
        print("No files found.")
    else:
        path = files[0]
        raw_points, found, missing = load_raw_markers(path)

        print(f"File: {path}")
        print(f"Raw markers found: {len(found)} / {len(RAW_MARKERS)}")
        if missing:
            print(f"Missing markers: {missing}")
        print(f"Filtered points array shape: {raw_points.shape}")
        print(f"(4 = X/Y/Z/residual | {raw_points.shape[1]} markers | {raw_points.shape[2]} frames)")