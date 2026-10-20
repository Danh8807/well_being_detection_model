import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base

LABELS = ["SWLS", "PA", "NA"]
SPEECH_COLUMNS = ["word_count", "words_per_second", "speech_fraction"]


def main(audio_name, layer):
    dataset = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    features = base.add_speech_statistics(base.load_feature_table(dataset[base.ID_COLUMN].tolist()))
    table = features.merge(dataset, on=base.ID_COLUMN, how="inner").reset_index(drop=True)
    text_all = base.get_text_embeddings(table)
    audio_all = base.load_audio_embeddings(table, audio_name)[:, layer]

    # Same exclusion as before: drop people with an empty transcript.
    keep = (table["word_count"] > 0).to_numpy()
    table = table[keep].reset_index(drop=True)
    text_all = text_all[keep]
    audio_all = audio_all[keep]

    is_train = (table["split"] == "train").to_numpy()
    speech = table[SPEECH_COLUMNS].to_numpy(dtype=float)
    targets = table[base.TARGET_COLUMNS].to_numpy(dtype=float)
    print("Train:", is_train.sum(), "| Test:", (~is_train).sum())

    feature_sets = {
        "text_only": text_all,
        "wavlm_audio_only": audio_all,
        "wavlm_all_features": np.hstack([audio_all, speech, text_all]),
    }

    y_train, y_test = targets[is_train], targets[~is_train]
    predictions = {}
    for name, matrix in feature_sets.items():
        model = base.build_ridge_model().fit(matrix[is_train], y_train)
        predictions[name] = model.predict(matrix[~is_train])
    mean_prediction = np.tile(y_train.mean(axis=0), (len(y_test), 1))

    rows = []
    for name, predicted in predictions.items():
        for i, label in enumerate(LABELS):
            lower, upper = base.bootstrap_interval_for_pearson(y_test[:, i], predicted[:, i])
            rows.append({
                "model": name, "target": label,
                "pearson_r": base.pearson_or_zero(y_test[:, i], predicted[:, i]),
                "ci_lower": lower, "ci_upper": upper,
                "r_squared": r2_score(y_test[:, i], predicted[:, i]),
                "mae": mean_absolute_error(y_test[:, i], predicted[:, i]),
            })
    for i, label in enumerate(LABELS):
        rows.append({
            "model": "mean_predictor", "target": label, "pearson_r": 0.0,
            "ci_lower": np.nan, "ci_upper": np.nan,
            "r_squared": r2_score(y_test[:, i], mean_prediction[:, i]),
            "mae": mean_absolute_error(y_test[:, i], mean_prediction[:, i]),
        })
    results = pd.DataFrame(rows)

    # Paired bootstrap: does adding audio change r compared with text only?
    rng = np.random.default_rng(base.RANDOM_SEED)
    n = len(y_test)
    differences = [[] for _ in LABELS]
    for _ in range(base.NUMBER_OF_BOOTSTRAP_RESAMPLES):
        idx = rng.integers(0, n, n)
        for i in range(3):
            r_both = base.pearson_or_zero(y_test[idx, i], predictions["wavlm_all_features"][idx, i])
            r_text = base.pearson_or_zero(y_test[idx, i], predictions["text_only"][idx, i])
            differences[i].append(r_both - r_text)

    print("\n", results.round(3).to_string(index=False))
    print("\nr(all_features) - r(text_only), paired bootstrap 95% CI:")
    for i, label in enumerate(LABELS):
        low, high = np.percentile(differences[i], [2.5, 97.5])
        observed = (base.pearson_or_zero(y_test[:, i], predictions["wavlm_all_features"][:, i])
                    - base.pearson_or_zero(y_test[:, i], predictions["text_only"][:, i]))
        print("  %-4s %+.3f  [%+.3f, %+.3f]" % (label, observed, low, high))

    out = base.OUTPUT_DIRECTORY / f"test_results_{audio_name}_layer{layer}.csv"
    results.to_csv(out, index=False)
    print("Saved", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_embedding", default="wavlm-base-plus")
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()
    main(args.audio_embedding, args.layer)