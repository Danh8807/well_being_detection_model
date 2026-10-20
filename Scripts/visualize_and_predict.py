import argparse
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base
import fusion_mlp as mlp

FIGURE_DIR = base.OUTPUT_DIRECTORY / "figures"
LABELS = ["SWLS", "PA", "NA"]
SPEECH_COLUMNS = ["word_count", "words_per_second", "speech_fraction"]
MODELS = ["ridge", "mlp"]
SEED = base.RANDOM_SEED


def load_data():
    dataset = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    features_table = base.load_feature_table(dataset[base.ID_COLUMN].tolist())
    features_table = base.add_speech_statistics(features_table)
    table = features_table.merge(dataset, on=base.ID_COLUMN).reset_index(drop=True)
    embeddings = base.get_text_embeddings(table)

    keep = (table["word_count"] > 0).to_numpy()
    table = table[keep].reset_index(drop=True)
    embeddings = embeddings[keep]

    acoustic_columns = [c for c in table.columns if c.startswith("acoustic_")]
    small = table[acoustic_columns + SPEECH_COLUMNS].to_numpy(dtype=float)
    features = np.hstack([small, embeddings])
    targets = table[base.TARGET_COLUMNS].to_numpy(dtype=float)
    is_train = (table["split"] == "train").to_numpy()

    n_acoustic = len(acoustic_columns)
    n_small = small.shape[1]
    blocks = {
        "acoustic": np.arange(0, n_acoustic),
        "speech stats": np.arange(n_acoustic, n_small),
        "text embedding": np.arange(n_small, features.shape[1]),
    }
    return table, features, targets, is_train, blocks


def fit_ridge(x, y):
    return base.build_ridge_model().fit(x, y)


def fit_predict(name, x_fit, y_fit, x_new, n_small):
    if name == "ridge":
        return fit_ridge(x_fit, y_fit).predict(x_new)
    acoustic_fit = x_fit[:, :n_small]
    text_fit = x_fit[:, n_small:]
    preprocessor = mlp.Preprocessor().fit(acoustic_fit, text_fit, y_fit)
    model = mlp.train_model(preprocessor, acoustic_fit, text_fit, y_fit, seed=SEED)
    return mlp.predict(model, preprocessor, x_new[:, :n_small], x_new[:, n_small:])


def out_of_fold(name, x, y, n_small):
    predicted = np.zeros_like(y)
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_index, held_index in splitter.split(y):
        predicted[held_index] = fit_predict(name, x[fit_index], y[fit_index], x[held_index], n_small)
        print("  %s fold done" % name)
    return predicted


def pearson_all(true_values, predicted):
    return [base.pearson_or_zero(true_values[:, i], predicted[:, i]) for i in range(3)]


def save(figure, name):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / name
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print("Saved", path)


def plot_scatter(true_values, predicted, suffix):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i in range(3):
        axis = axes[i]
        t = true_values[:, i]
        p = predicted[:, i]
        low = min(t.min(), p.min())
        high = max(t.max(), p.max())
        slope, intercept = np.polyfit(t, p, 1)
        axis.scatter(t, p, s=10, alpha=0.4)
        axis.plot([low, high], [low, high], "k--", label="y = x")
        axis.plot([low, high], [slope * low + intercept, slope * high + intercept], "r-",
                  label="fit (slope %.2f)" % slope)
        axis.set_title("%s: r = %.2f" % (LABELS[i], base.pearson_or_zero(t, p)))
        axis.set_xlabel("True score")
        axis.set_ylabel("Predicted score")
        axis.legend()
    figure.suptitle(suffix)
    save(figure, "01_pred_vs_true_%s.png" % suffix)


def plot_residuals(true_values, predicted, suffix):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i in range(3):
        residual = true_values[:, i] - predicted[:, i]
        axes[i].scatter(predicted[:, i], residual, s=10, alpha=0.4)
        axes[i].axhline(0, color="k")
        axes[i].set_title(LABELS[i] + ": residual (true - predicted)")
        axes[i].set_xlabel("Predicted score")
        axes[i].set_ylabel("Residual")
    figure.suptitle(suffix)
    save(figure, "02_residuals_%s.png" % suffix)


def plot_distributions(true_values, predicted, suffix):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for i in range(3):
        both = np.concatenate([true_values[:, i], predicted[:, i]])
        bins = np.linspace(both.min(), both.max(), 25)
        axes[i].hist(true_values[:, i], bins=bins, alpha=0.5, label="True")
        axes[i].hist(predicted[:, i], bins=bins, alpha=0.5, label="Predicted")
        axes[i].set_title("%s: SD true %.2f, predicted %.2f" % (LABELS[i], true_values[:, i].std(), predicted[:, i].std()))
        axes[i].legend()
    figure.suptitle(suffix)
    save(figure, "03_distributions_%s.png" % suffix)


def plot_error_by_word_count(table, true_values, predicted, suffix):
    words = table["word_count"].to_numpy()
    edges = np.unique(np.quantile(words, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
    group = np.clip(np.digitize(words, edges[1:-1]), 0, len(edges) - 2)
    names = []
    for b in range(len(edges) - 1):
        names.append("%d-%d" % (edges[b], edges[b + 1]))
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for i in range(3):
        errors = []
        for b in range(len(names)):
            mask = group == b
            errors.append(np.abs(true_values[mask, i] - predicted[mask, i]).mean())
        axis.plot(names, errors, marker="o", label=LABELS[i])
    axis.set_xlabel("Word count (5 equal-size groups)")
    axis.set_ylabel("MAE")
    axis.set_title(suffix)
    axis.legend()
    save(figure, "04_error_by_word_count_%s.png" % suffix)


def plot_model_comparison(x, y, n_small):
    scores = {}
    for name in MODELS:
        scores[name] = [[], [], []]
    splitter = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for fit_index, held_index in splitter.split(y):
        for name in MODELS:
            predicted = fit_predict(name, x[fit_index], y[fit_index], x[held_index], n_small)
            r_values = pearson_all(y[held_index], predicted)
            for i in range(3):
                scores[name][i].append(r_values[i])
        print("  comparison fold done")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    colors = {"ridge": "tab:blue", "mlp": "tab:orange"}
    for j, name in enumerate(MODELS):
        positions = np.arange(3) * 3 + j
        box = axis.boxplot(scores[name], positions=positions, widths=0.8, patch_artist=True)
        for patch in box["boxes"]:
            patch.set_facecolor(colors[name])
        axis.plot([], [], color=colors[name], linewidth=8, label=name)
        print(name, "mean r per target:", [round(float(np.mean(s)), 3) for s in scores[name]])
    axis.set_xticks(np.arange(3) * 3 + 0.5)
    axis.set_xticklabels(LABELS)
    axis.set_ylabel("Pearson r per fold (same folds for both models)")
    axis.legend()
    save(figure, "05_ridge_vs_mlp_folds.png")


def plot_learning_curve(x, y):
    fractions = [0.25, 0.5, 0.75, 1.0]
    results = {}
    for f in fractions:
        results[f] = [[], [], []]
    for repeat in range(2):
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        rng = np.random.default_rng(SEED + repeat)
        predicted = {}
        for f in fractions:
            predicted[f] = np.zeros_like(y)
        for fit_index, held_index in splitter.split(y):
            for f in fractions:
                size = int(f * len(fit_index))
                subset = rng.permutation(fit_index)[:size]
                predicted[f][held_index] = fit_ridge(x[subset], y[subset]).predict(x[held_index])
        for f in fractions:
            r_values = pearson_all(y, predicted[f])
            for i in range(3):
                results[f][i].append(r_values[i])
    sizes = [int(f * len(y) * 0.8) for f in fractions]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    for i in range(3):
        means = [np.mean(results[f][i]) for f in fractions]
        axis.plot(sizes, means, marker="o", label=LABELS[i])
    axis.set_xlabel("Training participants")
    axis.set_ylabel("Pearson r (CV, Ridge)")
    axis.legend()
    save(figure, "06_learning_curve_ridge.png")


def plot_block_importance(x, y, blocks):
    drops = {}
    for name in blocks:
        drops[name] = [[], [], []]
    rng = np.random.default_rng(SEED)
    for repeat in range(2):
        splitter = KFold(n_splits=5, shuffle=True, random_state=SEED + repeat)
        for fit_index, held_index in splitter.split(y):
            model = fit_ridge(x[fit_index], y[fit_index])
            held_x = x[held_index]
            held_y = y[held_index]
            baseline = pearson_all(held_y, model.predict(held_x))
            for name, columns in blocks.items():
                shuffled = held_x.copy()
                order = rng.permutation(len(held_x))
                shuffled[:, columns] = held_x[order][:, columns]
                after = pearson_all(held_y, model.predict(shuffled))
                for i in range(3):
                    drops[name][i].append(baseline[i] - after[i])
    figure, axis = plt.subplots(figsize=(8, 4.5))
    width = 0.25
    position = 0
    for name in blocks:
        values = [np.mean(drops[name][i]) for i in range(3)]
        axis.bar(np.arange(3) + position * width, values, width, label=name)
        position += 1
    axis.set_xticks(np.arange(3) + width)
    axis.set_xticklabels(LABELS)
    axis.set_ylabel("Drop in r when block is shuffled (Ridge)")
    axis.legend()
    save(figure, "07_block_importance_ridge.png")


def run_plots(on_test, model_choice):
    table, features, targets, is_train, blocks = load_data()
    n_small = int(blocks["text embedding"][0])
    if model_choice == "both":
        names = MODELS
    else:
        names = [model_choice]

    if on_test:
        mask = ~is_train
        split_name = "test"
    else:
        mask = is_train
        split_name = "cv_train"
    true_values = targets[mask]
    small_table = table[mask].reset_index(drop=True)

    for name in names:
        print("Computing predictions for", name)
        if on_test:
            predicted = fit_predict(name, features[is_train], targets[is_train], features[mask], n_small)
        else:
            predicted = out_of_fold(name, features[mask], targets[mask], n_small)
        suffix = name + "_" + split_name
        plot_scatter(true_values, predicted, suffix)
        plot_residuals(true_values, predicted, suffix)
        plot_distributions(true_values, predicted, suffix)
        plot_error_by_word_count(small_table, true_values, predicted, suffix)

    x_train = features[is_train]
    y_train = targets[is_train]
    if model_choice == "both":
        plot_model_comparison(x_train, y_train, n_small)
    plot_learning_curve(x_train, y_train)
    plot_block_importance(x_train, y_train, blocks)


def predict_people(identifiers, show_transcript=False, save_csv=False):
    table, features, targets, is_train, blocks = load_data()
    n_small = int(blocks["text embedding"][0])
    all_ids = table[base.ID_COLUMN].astype(str).tolist()
    test_positions = np.flatnonzero(~is_train)
    test_predictions = {}
    rows = []
    for identifier in identifiers:
        if str(identifier) not in all_ids:
            print("\n%s: not found (no features, empty transcript, or wrong ID)" % identifier)
            continue
        index = all_ids.index(str(identifier))
        predictions = {}
        if is_train[index]:
            keep = is_train.copy()
            keep[index] = False
            for name in MODELS:
                predictions[name] = fit_predict(name, features[keep], targets[keep], features[index:index + 1], n_small)[0]
            origin = "train (models refit without this person)"
        else:
            for name in MODELS:
                if name not in test_predictions:
                    print("Fitting", name, "on train...")
                    test_predictions[name] = fit_predict(name, features[is_train], targets[is_train],
                                                         features[test_positions], n_small)
                position = int(np.flatnonzero(test_positions == index)[0])
                predictions[name] = test_predictions[name][position]
            origin = "test"

        print("\n=== %s | %s | %d words ===" % (identifier, origin, table.loc[index, "word_count"]))
        row = {"ResponseId": identifier, "split": table.loc[index, "split"]}
        for i in range(3):
            ridge_value = predictions["ridge"][i]
            mlp_value = predictions["mlp"][i]
            print("  %-4s true %5.1f | ridge %5.1f (%+.1f) | mlp %5.1f (%+.1f)" % (
                LABELS[i], targets[index, i], ridge_value, ridge_value - targets[index, i],
                mlp_value, mlp_value - targets[index, i]))
            row[LABELS[i] + "_true"] = float(targets[index, i])
            row[LABELS[i] + "_ridge"] = round(float(ridge_value), 2)
            row[LABELS[i] + "_mlp"] = round(float(mlp_value), 2)
        rows.append(row)
        if show_transcript:
            print("  Transcript:", table.loc[index, "transcript"])

    if save_csv and len(rows) > 0:
        path = base.OUTPUT_DIRECTORY / "example_predictions.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print("\nSaved", path)


def pick_examples(count):
    table, _, targets, is_train, _ = load_data()
    test_positions = np.flatnonzero(~is_train)
    order = test_positions[np.argsort(targets[test_positions, 0])]
    chosen = order[np.linspace(0, len(order) - 1, count).astype(int)]
    return table.loc[chosen, base.ID_COLUMN].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    plots = commands.add_parser("plots")
    plots.add_argument("--on_test", action="store_true")
    plots.add_argument("--model", choices=["ridge", "mlp", "both"], default="both")

    predict = commands.add_parser("predict")
    predict.add_argument("--id", nargs="+", required=True)
    predict.add_argument("--show_transcript", action="store_true")

    examples = commands.add_parser("examples")
    examples.add_argument("--n", type=int, default=6)
    examples.add_argument("--show_transcript", action="store_true")

    args = parser.parse_args()
    if args.command == "plots":
        run_plots(args.on_test, args.model)
    elif args.command == "predict":
        predict_people(args.id, args.show_transcript)
    else:
        predict_people(pick_examples(args.n), args.show_transcript, save_csv=True)


if __name__ == "__main__":
    main()