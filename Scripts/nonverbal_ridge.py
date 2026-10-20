import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base


def summarize(sequence, mask):
    """Mean, std, and mean absolute change between consecutive kept frames."""
    kept = sequence[mask]
    if len(kept) < 2:
        return np.full(3 * sequence.shape[1], np.nan)
    change = np.abs(np.diff(kept, axis=0)).mean(axis=0)
    return np.concatenate([kept.mean(axis=0), kept.std(axis=0), change])


def main():
    data = np.load(base.OUTPUT_DIRECTORY / "sequences.npz", allow_pickle=True)
    is_train = data["split"] == "train"
    audio, face = data["audio"][is_train], data["face"][is_train]
    valid, face_ok = data["valid"][is_train], data["face_ok"][is_train]
    targets = data["targets"][is_train].astype(float)
    print("Train participants:", len(targets))

    audio_features = np.stack([summarize(audio[k], valid[k]) for k in range(len(audio))])
    face_features = np.stack([summarize(face[k], face_ok[k]) for k in range(len(face))])
    print("Audio feature dim:", audio_features.shape[1], "| face feature dim:", face_features.shape[1])

    feature_sets = {
        "nonverbal_audio": audio_features,
        "nonverbal_face": face_features,
        "nonverbal_audio+face": np.hstack([audio_features, face_features]),
    }
    rows = base.evaluate("mean_predictor", DummyRegressor(strategy="mean"), audio_features, targets)
    for name, matrix in feature_sets.items():
        rows += base.evaluate(name, base.build_ridge_model(), matrix, targets)

    results = pd.DataFrame(rows)
    results.to_csv(base.OUTPUT_DIRECTORY / "nonverbal_ridge_results.csv", index=False)
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()