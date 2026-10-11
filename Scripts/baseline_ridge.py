import json
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sentence_transformers import SentenceTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
FEATURE_DIRECTORY = OUTPUT_DIRECTORY / "features_raw"
EMBEDDING_CACHE_PATH = OUTPUT_DIRECTORY / "text_embeddings.npz"

ID_COLUMN = "ResponseId"
TARGET_COLUMNS: List[str] = ["swls_score", "positive_affect_score", "negative_affect_score"]
NUMBER_OF_FOLDS = 5
NUMBER_OF_REPEATS = 3
NUMBER_OF_BOOTSTRAP_RESAMPLES = 1000
RANDOM_SEED = 42


def load_feature_table(participant_identifiers: List[str]) -> pd.DataFrame:
    """Read the JSON file of every participant that has one (failed clips are skipped)."""
    rows: List[Dict] = []
    for participant_identifier in participant_identifiers:
        json_path = FEATURE_DIRECTORY / f"{participant_identifier}.json"
        if not json_path.exists():
            continue
        with open(json_path, "r", encoding="utf-8") as input_file:
            record = json.load(input_file)
        row = {
            ID_COLUMN: participant_identifier,
            "transcript": record["transcript"],
            "word_count": record["word_count"],
            "speech_duration_seconds": record["speech_duration_seconds"],
            "audio_duration_seconds": record["audio_duration_seconds"],
        }
        for feature_name, feature_value in record["acoustic_features"].items():
            row[f"acoustic_{feature_name}"] = feature_value
        rows.append(row)
    return pd.DataFrame(rows)


def add_speech_statistics(table: pd.DataFrame) -> pd.DataFrame:
    """Words per second of speech, and the fraction of the clip containing speech."""
    table = table.copy()
    table["words_per_second"] = table["word_count"] / table["speech_duration_seconds"].clip(lower=1e-3)
    table["speech_fraction"] = table["speech_duration_seconds"] / table["audio_duration_seconds"].clip(lower=1e-3)
    return table


def get_text_embeddings(table: pd.DataFrame) -> np.ndarray:
    """Sentence embeddings of the transcripts, cached on disk because encoding takes minutes."""
    identifiers = table[ID_COLUMN].to_numpy().astype(str)
    if EMBEDDING_CACHE_PATH.exists():
        cache = np.load(EMBEDDING_CACHE_PATH)
        if np.array_equal(cache["identifiers"], identifiers):
            return cache["embeddings"]
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    embeddings = model.encode(table["transcript"].fillna("").tolist(), batch_size=32, show_progress_bar=True)
    np.savez(EMBEDDING_CACHE_PATH, identifiers=identifiers, embeddings=embeddings)
    return embeddings


def concordance_correlation_coefficient(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Lin's CCC: penalizes low correlation and also differences in mean and scale."""
    covariance = np.mean((true_values - true_values.mean()) * (predicted_values - predicted_values.mean()))
    denominator = true_values.var() + predicted_values.var() + (true_values.mean() - predicted_values.mean()) ** 2
    return float(2.0 * covariance / denominator) if denominator > 0 else 0.0


def pearson_or_zero(true_values: np.ndarray, predicted_values: np.ndarray) -> float:
    """Pearson r, defined as 0 for a constant prediction (the mean predictor)."""
    if np.std(predicted_values) < 1e-12:
        return 0.0
    return float(pearsonr(true_values, predicted_values)[0])


def bootstrap_interval_for_pearson(true_values: np.ndarray, predicted_values: np.ndarray) -> List[float]:
    """95 percent bootstrap interval of Pearson r, resampling participants with replacement."""
    random_generator = np.random.default_rng(RANDOM_SEED)
    sample_count = len(true_values)
    resampled_values = []
    for _ in range(NUMBER_OF_BOOTSTRAP_RESAMPLES):
        indices = random_generator.integers(0, sample_count, sample_count)
        resampled_values.append(pearson_or_zero(true_values[indices], predicted_values[indices]))
    return [float(np.percentile(resampled_values, 2.5)), float(np.percentile(resampled_values, 97.5))]


def build_ridge_model():
    """
    Impute, standardize, then Ridge. RidgeCV picks the regularization strength
    on the training part of each fold only (alpha_per_target: one alpha per score).
    """
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RidgeCV(alphas=np.logspace(1, 5, 20), alpha_per_target=True),
    )


def out_of_fold_predictions(model, feature_matrix: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
    """
    Each participant is predicted by a model that never saw them. Repeated with
    different fold splits, then averaged to reduce the effect of one random split.
    """
    repeat_predictions = []
    for repeat_index in range(NUMBER_OF_REPEATS):
        splitter = KFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_SEED + repeat_index)
        repeat_predictions.append(cross_val_predict(model, feature_matrix, target_matrix, cv=splitter))
    return np.mean(repeat_predictions, axis=0)


def evaluate(name: str, model, feature_matrix: np.ndarray, target_matrix: np.ndarray) -> List[Dict]:
    predictions = out_of_fold_predictions(model, feature_matrix, target_matrix)
    rows = []
    for target_index, target_name in enumerate(TARGET_COLUMNS):
        true_values = target_matrix[:, target_index]
        predicted_values = predictions[:, target_index]
        lower, upper = bootstrap_interval_for_pearson(true_values, predicted_values)
        rows.append(
            {
                "feature_set": name,
                "target": target_name,
                "pearson_r": pearson_or_zero(true_values, predicted_values),
                "ci_lower": lower,
                "ci_upper": upper,
                "ccc": concordance_correlation_coefficient(true_values, predicted_values),
                "r_squared": float(r2_score(true_values, predicted_values)),
            }
        )
    return rows


def main() -> None:
    dataset_table = pd.read_csv(OUTPUT_DIRECTORY / "dataset_all.csv")
    feature_table = add_speech_statistics(load_feature_table(dataset_table[ID_COLUMN].tolist()))
    merged_table = feature_table.merge(dataset_table, on=ID_COLUMN, how="inner").reset_index(drop=True)
    print(f"Participants with features: {len(merged_table)} of {len(dataset_table)}")

    # Embeddings are computed for everyone (no labels involved) and cached.
    embedding_matrix = get_text_embeddings(merged_table)

    # Only the train split is used from here on.
    is_train = (merged_table["split"] == "train").to_numpy()
    acoustic_columns = [c for c in merged_table.columns if c.startswith("acoustic_")]
    speech_columns = ["word_count", "words_per_second", "speech_fraction"]

    acoustic_matrix = merged_table.loc[is_train, acoustic_columns].to_numpy(dtype=float)
    speech_matrix = merged_table.loc[is_train, speech_columns].to_numpy(dtype=float)
    text_matrix = embedding_matrix[is_train]
    target_matrix = merged_table.loc[is_train, TARGET_COLUMNS].to_numpy(dtype=float)
    print(f"Train participants used for cross-validation: {len(target_matrix)}")

    feature_sets = {
        "speech_statistics": speech_matrix,
        "acoustic_egemaps": acoustic_matrix,
        "text_embedding": text_matrix,
        "all_features": np.hstack([acoustic_matrix, speech_matrix, text_matrix]),
    }

    result_rows = evaluate("mean_predictor", DummyRegressor(strategy="mean"), acoustic_matrix, target_matrix)
    for name, feature_matrix in feature_sets.items():
        result_rows += evaluate(name, build_ridge_model(), feature_matrix, target_matrix)

    results_table = pd.DataFrame(result_rows)
    results_table.to_csv(OUTPUT_DIRECTORY / "baseline_ridge_results.csv", index=False)
    print(results_table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()