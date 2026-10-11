import pathlib

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import baseline_ridge as base

NUMBER_OF_PERMUTATIONS = 5


def main() -> None:
    dataset_table = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    feature_table = base.add_speech_statistics(base.load_feature_table(dataset_table[base.ID_COLUMN].tolist()))
    merged_table = feature_table.merge(dataset_table, on=base.ID_COLUMN).reset_index(drop=True)
    embeddings = base.get_text_embeddings(merged_table)

    is_train = (merged_table["split"] == "train").to_numpy()
    train_table = merged_table[is_train].reset_index(drop=True)
    train_embeddings = embeddings[is_train]
    swls = train_table["swls_score"].to_numpy(dtype=float)

    normalized = train_table["transcript"].fillna("").str.lower().str.strip()
    print(f"Duplicate transcripts in train: {int(normalized.duplicated().sum())}")
    print(f"Empty transcripts in train: {int((normalized.str.len() == 0).sum())}")
    print("Word count summary:")
    print(train_table["word_count"].describe().round(1).to_string())

    model = make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(1, 5, 20)))
    splitter = KFold(n_splits=5, shuffle=True, random_state=0)
    random_generator = np.random.default_rng(0)
    print("\nPermutation test (labels shuffled, expect r near 0):")
    for permutation_index in range(NUMBER_OF_PERMUTATIONS):
        shuffled = random_generator.permutation(swls)
        predictions = cross_val_predict(model, train_embeddings, shuffled, cv=splitter)
        print(f"  permutation {permutation_index + 1}: r = {pearsonr(shuffled, predictions)[0]:.3f}")

    ordered = train_table.sort_values("swls_score")
    for label, group in (("LOWEST SWLS", ordered.head(3)), ("HIGHEST SWLS", ordered.tail(3))):
        print(f"\n{label}")
        for _, row in group.iterrows():
            print(f"  score {row['swls_score']:.0f}: {str(row['transcript'])[:300]}")


if __name__ == "__main__":
    main()