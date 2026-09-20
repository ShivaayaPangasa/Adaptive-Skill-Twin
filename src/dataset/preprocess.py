import numpy as np
import ezc3d
from marker_filter import RAW_MARKERS

PELVIS_MARKERS = ["LASI", "RASI", "LPSI", "RPSI"]
SCALE_MARKERS = ("C7", "LASI")  # torso length as a body-size reference
FIXED_LENGTH = 100
SPEED_PERCENTILE = 60
PAD_FRAMES = 5


def _strip_prefix(label):
    """Vicon sometimes prefixes marker labels with a subject name when
    multiple people are captured in one trial, e.g. 'B0367:LFHD'.
    This strips that prefix so matching works either way."""
    return label.split(":")[-1] if ":" in label else label


def load_raw_markers(path):
    c3d = ezc3d.c3d(str(path))
    labels = c3d['parameters']['POINT']['LABELS']['value']
    points = c3d['data']['points']

    clean_labels = [_strip_prefix(l) for l in labels]

    idx = []
    missing = []
    for m in RAW_MARKERS:
        if m in clean_labels:
            idx.append(clean_labels.index(m))
        else:
            missing.append(m)

    if missing:
        raise ValueError(
            f"Missing markers: {missing}\n"
            f"  Actual labels in file (first 20): {labels[:20]}"
        )

    xyz = points[:3, idx, :].transpose(2, 1, 0)       # (frames, 39, 3)
    residual = points[3, idx, :].T                     # (frames, 39)
    return xyz, residual


def interpolate_missing(xyz, residual):
    """Fill occluded-marker gaps (residual < 0) via linear interpolation over time."""
    n_frames, n_markers, _ = xyz.shape
    for m in range(n_markers):
        missing = residual[:, m] < 0
        if missing.any() and not missing.all():
            valid = np.where(~missing)[0]
            for ax in range(3):
                xyz[missing, m, ax] = np.interp(np.where(missing)[0], valid, xyz[valid, m, ax])
    return xyz


def center_on_pelvis(xyz):
    idx = [RAW_MARKERS.index(m) for m in PELVIS_MARKERS]
    return xyz - xyz[:, idx, :].mean(axis=1, keepdims=True)


def normalize_scale(xyz):
    i, j = RAW_MARKERS.index(SCALE_MARKERS[0]), RAW_MARKERS.index(SCALE_MARKERS[1])
    ref = np.linalg.norm(xyz[:, i, :] - xyz[:, j, :], axis=1).mean()
    return xyz / (ref if ref > 1e-6 else 1.0)


def trim_idle_frames(xyz):
    """Keep only the active-motion window, trimming idle setup/recovery frames."""
    speed = np.linalg.norm(np.diff(xyz, axis=0), axis=2).sum(axis=1)
    active = np.where(speed > np.percentile(speed, SPEED_PERCENTILE))[0]
    if len(active) == 0:
        return xyz
    start = max(0, active[0] - PAD_FRAMES)
    end = min(len(xyz), active[-1] + PAD_FRAMES + 2)
    return xyz[start:end]


def resample_time(xyz, target_len=FIXED_LENGTH):
    """Resample to a fixed frame count so every sample is the same tensor shape."""
    old_t = np.linspace(0, 1, xyz.shape[0])
    new_t = np.linspace(0, 1, target_len)
    out = np.zeros((target_len, xyz.shape[1], xyz.shape[2]))
    for m in range(xyz.shape[1]):
        for ax in range(3):
            out[:, m, ax] = np.interp(new_t, old_t, xyz[:, m, ax])
    return out


def preprocess_file(path):
    xyz, residual = load_raw_markers(path)
    xyz = interpolate_missing(xyz, residual)
    xyz = center_on_pelvis(xyz)
    xyz = normalize_scale(xyz)
    xyz = trim_idle_frames(xyz)
    xyz = resample_time(xyz)
    return xyz.astype(np.float32)  # shape: (100, 39, 3)