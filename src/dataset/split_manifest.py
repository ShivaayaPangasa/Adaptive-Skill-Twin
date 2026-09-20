import pandas as pd
from pathlib import Path
import random

MANIFEST_PATH = Path("data/manifest.csv")
SPLIT_MANIFEST_PATH = Path("data/manifest_split.csv")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining ~0.15 goes to test
SEED = 42


def split_participants(participants, train_frac, val_frac, seed):
    participants = sorted(participants)  # sort first so shuffle is reproducible
    rng = random.Random(seed)
    rng.shuffle(participants)

    n = len(participants)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_ids = set(participants[:n_train])
    val_ids = set(participants[n_train:n_train + n_val])
    test_ids = set(participants[n_train + n_val:])
    return train_ids, val_ids, test_ids


def assign_split(participant_id, train_ids, val_ids, test_ids):
    if participant_id in train_ids:
        return "train"
    elif participant_id in val_ids:
        return "val"
    elif participant_id in test_ids:
        return "test"
    return "unknown"


if __name__ == "__main__":
    df = pd.read_csv(MANIFEST_PATH)

    valid_df = df[df["participant"].notna()].copy()
    dropped = len(df) - len(valid_df)
    if dropped:
        print(f"Dropping {dropped} rows with unparsed participant IDs (check these manually).")

    participants = valid_df["participant"].unique().tolist()
    print(f"Total participants: {len(participants)}")

    train_ids, val_ids, test_ids = split_participants(participants, TRAIN_FRAC, VAL_FRAC, SEED)

    print(f"Train participants: {len(train_ids)}")
    print(f"Val participants:   {len(val_ids)}")
    print(f"Test participants:  {len(test_ids)}")

    # Sanity check: this is the single most important guarantee in this whole script
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
    print("Confirmed: no participant appears in more than one split.")

    valid_df["split"] = valid_df["participant"].apply(
        lambda p: assign_split(p, train_ids, val_ids, test_ids)
    )

    print("\nFile counts per split:")
    print(valid_df["split"].value_counts())

    valid_df.to_csv(SPLIT_MANIFEST_PATH, index=False)
    print(f"\nSplit manifest saved to {SPLIT_MANIFEST_PATH}")