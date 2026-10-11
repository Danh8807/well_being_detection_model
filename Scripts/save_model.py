import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base

MODEL_DIRECTORY = base.OUTPUT_DIRECTORY / "models"


def main() -> None:
    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataset_table = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    feature_table = base.add_speech_statistics(base.load_feature_table(dataset_table[base.ID_COLUMN].tolist()))
    merged_table = feature_table.merge(dataset_table, on=base.ID_COLUMN).reset_index(drop=True)
    embeddings = base.get_text_embeddings(merged_table)

    has_speech = (merged_table["word_count"] > 0).to_numpy()
    merged_table, embeddings = merged_table[has_speech].reset_index(drop=True), embeddings[has_speech]
    is_train = (merged_table["split"] == "train").to_numpy()

    acoustic_columns = [c for c in merged_table.columns if c.startswith("acoustic_")]
    speech_columns = ["word_count", "words_per_second", "speech_fraction"]
    acoustic_matrix = merged_table[acoustic_columns + speech_columns].to_numpy(dtype=float)
    targets = merged_table[base.TARGET_COLUMNS].to_numpy(dtype=float)

    models_to_save = {
        "ridge_all_features": np.hstack([acoustic_matrix, embeddings]),
        "ridge_text_only": embeddings,
    }
    for name, matrix in models_to_save.items():
        model = base.build_ridge_model()
        model.fit(matrix[is_train], targets[is_train])
        save_path = MODEL_DIRECTORY / f"{name}.joblib"
        joblib.dump(
            {
                "model": model,
                "target_columns": base.TARGET_COLUMNS,
                "input_feature_order": "88 eGeMAPS, 3 speech statistics, 768 text embedding"
                if name == "ridge_all_features" else "768 text embedding",
                "text_encoder": "sentence-transformers/all-mpnet-base-v2",
            },
            save_path,
        )

        # Reload and compare predictions to confirm the saved file works.
        reloaded = joblib.load(save_path)["model"]
        assert np.allclose(reloaded.predict(matrix[:5]), model.predict(matrix[:5]))
        print(f"Saved {save_path.name}: {matrix.shape[1]} input features, "
              f"{save_path.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()