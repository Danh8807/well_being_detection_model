import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base  # reuses loading, embedding and metric functions


def evaluate_on_test(name, model, train_x, train_y, test_x, test_y):
    """Fit on train, predict test, and return one result row per target with bootstrap CI."""
    model.fit(train_x, train_y)
    predictions = model.predict(test_x)
    rows = []
    for index, target_name in enumerate(base.TARGET_COLUMNS):
        true_values, predicted_values = test_y[:, index], predictions[:, index]
        lower, upper = base.bootstrap_interval_for_pearson(true_values, predicted_values)
        rows.append(
            {
                "model": name,
                "target": target_name,
                "pearson_r": base.pearson_or_zero(true_values, predicted_values),
                "ci_lower": lower,
                "ci_upper": upper,
                "ccc": base.concordance_correlation_coefficient(true_values, predicted_values),
                "r_squared": float(r2_score(true_values, predicted_values)),
                "mae": float(mean_absolute_error(true_values, predicted_values)),
            }
        )
    return rows, predictions


def main() -> None:
    dataset_table = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    feature_table = base.add_speech_statistics(base.load_feature_table(dataset_table[base.ID_COLUMN].tolist()))
    merged_table = feature_table.merge(dataset_table, on=base.ID_COLUMN).reset_index(drop=True)
    embeddings = base.get_text_embeddings(merged_table)

    # Remove participants whose video has no recognized speech.
    has_speech = (merged_table["word_count"] > 0).to_numpy()
    merged_table, embeddings = merged_table[has_speech].reset_index(drop=True), embeddings[has_speech]

    is_train = (merged_table["split"] == "train").to_numpy()
    is_test = ~is_train
    print(f"Train: {is_train.sum()}  Test: {is_test.sum()}")

    acoustic_columns = [c for c in merged_table.columns if c.startswith("acoustic_")]
    speech_columns = ["word_count", "words_per_second", "speech_fraction"]
    acoustic_all = merged_table[acoustic_columns + speech_columns].to_numpy(dtype=float)
    feature_sets = {
        "ridge_all_features (PRIMARY)": np.hstack([acoustic_all, embeddings]),
        "ridge_text_only": embeddings,
        "ridge_acoustic_only": acoustic_all,
    }
    targets = merged_table[base.TARGET_COLUMNS].to_numpy(dtype=float)

    result_rows = []
    rows, _ = evaluate_on_test(
        "mean_predictor", DummyRegressor(strategy="mean"),
        acoustic_all[is_train], targets[is_train], acoustic_all[is_test], targets[is_test],
    )
    result_rows += rows

    for name, matrix in feature_sets.items():
        rows, predictions = evaluate_on_test(
            name, base.build_ridge_model(),
            matrix[is_train], targets[is_train], matrix[is_test], targets[is_test],
        )
        result_rows += rows
        if "PRIMARY" in name:
            saved = merged_table.loc[is_test, [base.ID_COLUMN]].copy()
            for index, target_name in enumerate(base.TARGET_COLUMNS):
                saved[f"true_{target_name}"] = targets[is_test][:, index]
                saved[f"predicted_{target_name}"] = predictions[:, index]
            saved.to_csv(base.OUTPUT_DIRECTORY / "test_predictions_primary.csv", index=False)

    results_table = pd.DataFrame(result_rows)
    results_table.to_csv(base.OUTPUT_DIRECTORY / "test_results.csv", index=False)
    print(results_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()