import os
import sys
import csv
from collections import defaultdict

with open("FingerSure_Dataset/metadata.csv", "r", encoding="utf-8") as f:
    csv_rows = list(csv.DictReader(f))

# Group by lower-case participant
by_p_lower = defaultdict(list)
for idx, r in enumerate(csv_rows, start=2):
    p_lower = r['participant_id'].strip().lower()
    by_p_lower[p_lower].append((idx, r))

# Let's inspect participants with multiple case variants
for p_lower in sorted(by_p_lower.keys()):
    rows = by_p_lower[p_lower]
    variants = set(r['participant_id'] for idx, r in rows)
    if len(variants) > 1 or p_lower in ['kusuma', 'bhushan', 'ashitta', 'bharthi']:
        print(f"\nParticipant lower '{p_lower}' (variants: {variants}) - {len(rows)} entries:")
        for idx, r in rows:
            print(f"  Line {idx:3d}: pid='{r['participant_id']}', hand={r['hand']}, finger={r['finger']}, cond={r['condition']}, file='{r['filename']}', time={r['capture_timestamp']}")

