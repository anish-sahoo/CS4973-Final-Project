
import os
import csv
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import config as config

ANNOT_DIR = os.path.join(config.DATA_DIR, "MPIIGaze", "Annotation Subset")
IMG_ROOT = os.path.join(config.DATA_DIR, "MPIIGaze", "Data", "Original")
OUTPUT = os.path.join(config.DATA_DIR, "mpiigaze_landmarks.csv")

rows = []

for fname in sorted(os.listdir(ANNOT_DIR)):
    if not fname.endswith(".txt"):
        continue
    
    with open(os.path.join(ANNOT_DIR, fname), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 17:
                continue
            
            rel_path = parts[0]  # e.g. day13/0203.jpg
            coords = list(map(float, parts[1:]))

            # absolute path
            img_path = os.path.join(IMG_ROOT, rel_path)

            rows.append([img_path] + coords)

# write
with open(OUTPUT, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["image_path"]
    for i in range(1, 9):
        header += [f"x{i}", f"y{i}"]
    writer.writerow(header)
    writer.writerows(rows)

print("Wrote", len(rows), "rows to", OUTPUT)
