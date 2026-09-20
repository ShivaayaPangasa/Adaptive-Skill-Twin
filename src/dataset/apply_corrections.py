import csv
from pathlib import Path

MANIFEST_PATH = Path("data/manifest_split.csv")

# Official correction notice from the Figshare dataset page (posted 02/03/2023):
# these 3 files were originally saved with swapped/incorrect exercise-trial codes.
FILENAME_CORRECTIONS = {
    "2017-03-03-B0388-S05-E01-T02.c3d": "2017-03-03-B0388-S05-E02-T01.c3d",
    "2017-03-03-B0388-S05-E02-T01.c3d": "2017-03-03-B0388-S05-E01-T02.c3d",
    "2017-03-07-B0396-S03-E02-T01.c3d": "2017-03-07-B0396-S03-E01-T03.c3d",
}


def corrected_fields(filename):
    """Given a corrected filename, re-derive exercise/trial from it,
    normalized to strip any zero-padding."""
    parts = filename.replace(".c3d", "").split("-")
    exercise = str(int(parts[-2].lstrip("Ee")))
    trial = str(int(parts[-1].lstrip("Tt")))
    return exercise, trial


if __name__ == "__main__":
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fixed_count = 0
    for row in rows:
        if row["filename"] in FILENAME_CORRECTIONS:
            correct_name = FILENAME_CORRECTIONS[row["filename"]]
            new_exercise, new_trial = corrected_fields(correct_name)
            print(f"Correcting {row['filename']} -> exercise {row['exercise']}->{new_exercise}, "
                  f"trial {row['trial']}->{new_trial}")
            row["exercise"] = new_exercise
            row["trial"] = new_trial
            fixed_count += 1

    print(f"\nCorrected {fixed_count} row(s).")

    fieldnames = list(rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest updated in place: {MANIFEST_PATH}")