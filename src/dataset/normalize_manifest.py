import csv
from pathlib import Path

MANIFEST_PATH = Path("data/manifest_split.csv")


def normalize_numeric(value):
    """Strip leading zeros so '01' and '1' become the same value."""
    if value is None or value == "":
        return value
    return str(int(value))


if __name__ == "__main__":
    with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        for field in ("session", "exercise", "trial"):
            row[field] = normalize_numeric(row[field])

    fieldnames = list(rows[0].keys())
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Normalized session/exercise/trial fields in {MANIFEST_PATH}")