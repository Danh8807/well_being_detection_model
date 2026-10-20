import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base
from nonverbal_ridge import summarize

LOW_MAX, MID_MAX = 10, 20      # classes: <=10 | 11-20 | >=21 (fixed before looking at results)
CLASS_NAMES = ["low (<=10)", "middle (11-20)", "high (>=21)"]


def text_embeddings_for(ids):
    dataset = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    features = base.add_speech_statistics(base.load_feature_table(dataset[base.ID_COLUMN].tolist()))
    merged = features.merge(dataset, on=base.ID_COLUMN, how="inner").reset_index(drop=True)
    embeddings = base.get_text_embeddings(merged)
    position = {str(i): k for k, i in enumerate(merged[base.ID_COLUMN])}
    return embeddings[[position[str(i)] for i in ids]]


def bootstrap_ci(y, proba, predicted):
    rng = np.random.default_rng(base.RANDOM_SEED)
    scores = {"balanced_accuracy": [], "macro_auc": []}
    for _ in range(500):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 3:
            continue
        scores["balanced_accuracy"].append(balanced_accuracy_score(y[idx], predicted[idx]))
        scores["macro_auc"].append(roc_auc_score(y[idx], proba[idx], multi_class="ovr", average="macro"))
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in scores.items()}


def main():
    data = np.load(base.OUTPUT_DIRECTORY / "sequences.npz", allow_pickle=True)
    is_train = data["split"] == "train"
    ids = data["ids"][is_train]
    swls = data["targets"][is_train][:, 0]
    y = np.where(swls <= LOW_MAX, 0, np.where(swls <= MID_MAX, 1, 2))
    counts = np.bincount(y, minlength=3)
    print("Train participants:", len(y))
    for name, n in zip(CLASS_NAMES, counts):
        print("  %-16s %4d (%.1f%%)" % (name, n, 100 * n / len(y)))
    print("Majority-class accuracy (do NOT report as skill): %.3f" % (counts.max() / len(y)))

    audio, face = data["audio"][is_train], data["face"][is_train]
    valid, face_ok = data["valid"][is_train], data["face_ok"][is_train]
    audio_f = np.stack([summarize(audio[k], valid[k]) for k in range(len(y))])
    face_f = np.stack([summarize(face[k], face_ok[k]) for k in range(len(y))])
    feature_sets = {
        "nonverbal_audio": audio_f,
        "nonverbal_face": face_f,
        "nonverbal_audio+face": np.hstack([audio_f, face_f]),
        "text_embedding (reference)": text_embeddings_for(ids),
    }

    folds = StratifiedKFold(5, shuffle=True, random_state=base.RANDOM_SEED)
    rows = []
    for name, matrix in feature_sets.items():
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              LogisticRegression(C=0.01, class_weight="balanced", max_iter=2000))
        proba = cross_val_predict(model, matrix, y, cv=folds, method="predict_proba")
        predicted = proba.argmax(axis=1)
        ci = bootstrap_ci(y, proba, predicted)
        rows.append({
            "feature_set": name,
            "balanced_acc": balanced_accuracy_score(y, predicted),
            "bacc_lo": ci["balanced_accuracy"][0], "bacc_hi": ci["balanced_accuracy"][1],
            "macro_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
            "auc_lo": ci["macro_auc"][0], "auc_hi": ci["macro_auc"][1],
            "macro_f1": f1_score(y, predicted, average="macro"),
        })
    results = pd.DataFrame(rows)
    results.to_csv(base.OUTPUT_DIRECTORY / "nonverbal_classify_results.csv", index=False)
    print("\nChance: balanced accuracy 0.333, macro AUC 0.500")
    print(results.round(3).to_string(index=False))


if __name__ == "__main__":
    main()