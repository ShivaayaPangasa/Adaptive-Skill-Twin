import csv
from collections import Counter, defaultdict
from pathlib import Path

MANIFEST_PATH = Path("data/manifest_split.csv")

with open(MANIFEST_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

exercise_counts = Counter(r["exercise"] for r in rows)
print("Overall exercise code distribution:")
for code, count in sorted(exercise_counts.items()):
    print(f"  E{code}: {count} files")

per_participant = defaultdict(set)
for r in rows:
    per_participant[r["participant"]].add(r["exercise"])

missing_codes = {p: codes for p, codes in per_participant.items() if len(codes) < 4}
print(f"\nParticipants with fewer than 4 distinct exercise codes: {len(missing_codes)}")
for p, codes in list(missing_codes.items())[:10]:
    print(f"  {p}: has codes {sorted(codes)}")