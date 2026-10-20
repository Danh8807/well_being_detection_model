# Scripts/check_sequences.py
import pathlib

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
LANDMARKS = OUTPUT_DIRECTORY / "landmarks_mp_raw"
AUDIO = OUTPUT_DIRECTORY / "audio_lld_raw"
ID_COLUMN = "ResponseId"

dataset = pd.read_csv(OUTPUT_DIRECTORY / "dataset_all.csv")
landmark_ids = {p.stem for p in LANDMARKS.glob("*.npy")}
audio_ids = {p.stem for p in AUDIO.glob("*.npy")}
dataset_ids = set(dataset[ID_COLUMN].astype(str))

both = landmark_ids & audio_ids & dataset_ids
print("In dataset:", len(dataset_ids), "| landmarks:", len(landmark_ids), "| audio:", len(audio_ids), "| both:", len(both))
print("Landmarks but no audio:", sorted(landmark_ids - audio_ids))
print("Audio but no landmarks:", sorted(audio_ids - landmark_ids))

rows = []
for identifier in sorted(both):
    face = np.load(LANDMARKS / f"{identifier}.npy")
    voice = np.load(AUDIO / f"{identifier}.npy")
    rows.append({
        ID_COLUMN: identifier,
        "n_face": len(face),
        "n_audio": len(voice),
        "face_detected": float((~np.isnan(face[:, 0])).mean()),
        "audio_nan": float(np.isnan(voice).mean()),
    })
report = pd.DataFrame(rows)
report["length_gap"] = (report["n_face"] - report["n_audio"]).abs()
report.to_csv(OUTPUT_DIRECTORY / "sequence_check.csv", index=False)

print("\nSeconds per person:", report["n_face"].describe()[["min", "50%", "max"]].round(0).to_dict())
print("Length gap (face vs audio), seconds:", report["length_gap"].value_counts().sort_index().to_dict())
print("Face detected: mean %.3f | people below 0.9: %d | below 0.5: %d" % (
    report["face_detected"].mean(), (report["face_detected"] < 0.9).sum(), (report["face_detected"] < 0.5).sum()))
print("Audio NaN fraction: mean %.3f | max %.3f" % (report["audio_nan"].mean(), report["audio_nan"].max()))