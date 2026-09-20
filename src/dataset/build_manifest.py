import re
import ezc3d
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")
MANIFEST_PATH = Path("data/manifest.csv")

FILENAME_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})-"
    r"(?P<participant>[A-Za-z0-9]+)-"
    r"S(?P<session>\d+)-"
    r"E(?P<exercise>\d+)-"
    r"T(?P<trial>\d+)\.c3d$",
    re.IGNORECASE,
)


def parse_filename(filename):
    """Extract participant/session/exercise/trial from a filename.
    Returns None fields (not a crash) if a file doesn't match the
    expected pattern, so one oddly-named file doesn't stop the build.
    Numeric fields are normalized (e.g. '01' -> '1') so codes are
    consistent regardless of zero-padding in the original filename."""
    match = FILENAME_PATTERN.match(filename)
    if not match:
        return {"date": None, "participant": None,
                 "session": None, "exercise": None, "trial": None}
    fields = match.groupdict()
    for key in ("session", "exercise", "trial"):
        fields[key] = str(int(fields[key]))
    return fields


def get_c3d_stats(path):
    """Load just enough of the file to record its basic stats."""
    try:
        c3d = ezc3d.c3d(str(path))
        sampling_rate = c3d['parameters']['POINT']['RATE']['value'][0]
        n_frames = c3d['data']['points'].shape[2]
        n_points = c3d['data']['points'].shape[1]
        return {
            "sampling_rate": sampling_rate,
            "n_frames": n_frames,
            "n_points": n_points,
            "duration_sec": n_frames / sampling_rate if sampling_rate else None,
            "load_error": None,
        }
    except Exception as e:
        return {"sampling_rate": None, "n_frames": None, "n_points": None,
                 "duration_sec": None, "load_error": str(e)}


def build_manifest(data_dir):
    rows = []
    files = sorted(data_dir.rglob("*.c3d"))
    print(f"Scanning {len(files)} C3D files...")

    for i, path in enumerate(files, start=1):
        parsed = parse_filename(path.name)
        stats = get_c3d_stats(path)
        rows.append({"filepath": str(path), "filename": path.name, **parsed, **stats})
        if i % 50 == 0:
            print(f"  processed {i}/{len(files)}")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_manifest(DATA_DIR)

    print(f"\nTotal files: {len(df)}")
    print(f"Unique participants found: {df['participant'].nunique()}")
    print(f"Files with unparsed filenames: {df['participant'].isna().sum()}")
    print(f"Files that failed to load: {df['load_error'].notna().sum()}")

    df.to_csv(MANIFEST_PATH, index=False)
    print(f"\nManifest saved to {MANIFEST_PATH}")