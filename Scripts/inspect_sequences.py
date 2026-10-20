# Scripts/inspect_sequences.py
import pathlib

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

report = pd.read_csv(OUTPUT_DIRECTORY / "sequence_check.csv")
columns = pd.read_csv(OUTPUT_DIRECTORY / "audio_lld_columns.csv", header=None)[0].tolist()

print("Large length gap:")
print(report[report["length_gap"] > 1].to_string(index=False))

print("\nFace detected below 0.9:")
print(report[report["face_detected"] < 0.9].sort_values("face_detected").to_string(index=False))

# Are unvoiced frames stored as zeros? Count exact zeros per descriptor.
zero_share = np.zeros(len(columns))
count = 0
for path in sorted((OUTPUT_DIRECTORY / "audio_lld_raw").glob("*.npy"))[:200]:
    values = np.load(path)
    zero_share += (values == 0).mean(axis=0)
    count += 1
print("\nShare of exactly-zero seconds per descriptor (first 200 people):")
for name, share in sorted(zip(columns, zero_share / count), key=lambda t: -t[1]):
    print("  %-45s %.3f" % (name, share))