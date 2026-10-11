import argparse
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base


BRANCH_DIMENSION = 64
HIDDEN_DIMENSION = 64
DROPOUT = 0.3
MODALITY_DROPOUT = 0.2
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
BATCH_SIZE = 64
MAXIMUM_EPOCHS = 150
PATIENCE = 15
VALIDATION_FRACTION = 0.1
RANDOM_SEED = 42


class FusionMLP(nn.Module):
    def __init__(self, acoustic_dimension: int, text_dimension: int) -> None:
        super().__init__()
        self.acoustic_branch = nn.Sequential(
            nn.Linear(acoustic_dimension, BRANCH_DIMENSION), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.text_branch = nn.Sequential(
            nn.Linear(text_dimension, BRANCH_DIMENSION), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.shared_layer = nn.Sequential(
            nn.Linear(2 * BRANCH_DIMENSION, HIDDEN_DIMENSION), nn.ReLU(), nn.Dropout(DROPOUT)
        )
        self.output_layer = nn.Linear(HIDDEN_DIMENSION, len(base.TARGET_COLUMNS))

    def forward(self, acoustic: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        acoustic_hidden = self.acoustic_branch(acoustic)
        text_hidden = self.text_branch(text)
        if self.training and MODALITY_DROPOUT > 0:
            # One random number per person: a low value hides acoustic, a value in
            # the next band hides text, so both are never hidden together.
            draw = torch.rand(acoustic.shape[0], 1)
            acoustic_hidden = acoustic_hidden * (draw >= MODALITY_DROPOUT).float()
            text_hidden = text_hidden * ((draw < MODALITY_DROPOUT) | (draw >= 2 * MODALITY_DROPOUT)).float()
        return self.output_layer(self.shared_layer(torch.cat([acoustic_hidden, text_hidden], dim=1)))


def load_arrays():
    """Return acoustic matrix, text embeddings, targets, split mask, ids (empty transcripts removed)."""
    dataset_table = pd.read_csv(base.OUTPUT_DIRECTORY / "dataset_all.csv")
    feature_table = base.add_speech_statistics(base.load_feature_table(dataset_table[base.ID_COLUMN].tolist()))
    merged_table = feature_table.merge(dataset_table, on=base.ID_COLUMN).reset_index(drop=True)
    embeddings = base.get_text_embeddings(merged_table)

    has_speech = (merged_table["word_count"] > 0).to_numpy()
    merged_table, embeddings = merged_table[has_speech].reset_index(drop=True), embeddings[has_speech]

    acoustic_columns = [c for c in merged_table.columns if c.startswith("acoustic_")]
    speech_columns = ["word_count", "words_per_second", "speech_fraction"]
    acoustic_matrix = merged_table[acoustic_columns + speech_columns].to_numpy(dtype=float)
    targets = merged_table[base.TARGET_COLUMNS].to_numpy(dtype=float)
    is_train = (merged_table["split"] == "train").to_numpy()
    return acoustic_matrix, embeddings, targets, is_train, merged_table[base.ID_COLUMN]


class Preprocessor:
    """Imputation and standardization fitted on training data only."""

    def fit(self, acoustic_train, text_train, target_train):
        self.acoustic_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit(acoustic_train)
        self.text_scaler = StandardScaler().fit(text_train)
        self.target_mean = target_train.mean(axis=0)
        self.target_std = target_train.std(axis=0)
        return self

    def inputs(self, acoustic, text):
        return (
            torch.tensor(self.acoustic_pipeline.transform(acoustic), dtype=torch.float32),
            torch.tensor(self.text_scaler.transform(text), dtype=torch.float32),
        )

    def scaled_targets(self, targets):
        return torch.tensor((targets - self.target_mean) / self.target_std, dtype=torch.float32)

    def original_scale(self, scaled_predictions):
        return scaled_predictions * self.target_std + self.target_mean


def train_model(preprocessor, acoustic_train, text_train, target_train, seed, verbose=False):
    """Train with early stopping on a small validation part of the training data."""
    torch.manual_seed(seed)
    acoustic_all, text_all = preprocessor.inputs(acoustic_train, text_train)
    target_all = preprocessor.scaled_targets(target_train)

    shuffled = np.random.default_rng(seed).permutation(len(target_all))
    validation_count = int(VALIDATION_FRACTION * len(shuffled))
    validation_index, fit_index = shuffled[:validation_count], shuffled[validation_count:]

    model = FusionMLP(acoustic_all.shape[1], text_all.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.MSELoss()

    best_validation_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    for epoch in range(MAXIMUM_EPOCHS):
        model.train()
        order = torch.from_numpy(np.random.default_rng(seed + epoch).permutation(fit_index))
        running_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_function(model(acoustic_all[batch], text_all[batch]), target_all[batch])
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(batch)
        train_loss = running_loss / len(order)

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(acoustic_all[validation_index], text_all[validation_index]), target_all[validation_index]
            ).item()
        if verbose and epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  train loss {train_loss:.3f}  validation loss {validation_loss:.3f}")

        if validation_loss < best_validation_loss:
            best_validation_loss, epochs_without_improvement = validation_loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    if verbose:
        print(f"  stopped at epoch {epoch}, best validation loss {best_validation_loss:.3f}")
    return model


def predict(model, preprocessor, acoustic, text):
    acoustic_tensor, text_tensor = preprocessor.inputs(acoustic, text)
    with torch.no_grad():
        return preprocessor.original_scale(model(acoustic_tensor, text_tensor).numpy())


def score(name, targets, predictions):
    rows = []
    for index, target_name in enumerate(base.TARGET_COLUMNS):
        true_values, predicted_values = targets[:, index], predictions[:, index]
        lower, upper = base.bootstrap_interval_for_pearson(true_values, predicted_values)
        rows.append({
            "model": name, "target": target_name,
            "pearson_r": base.pearson_or_zero(true_values, predicted_values),
            "ci_lower": lower, "ci_upper": upper,
            "ccc": base.concordance_correlation_coefficient(true_values, predicted_values),
            "r_squared": float(r2_score(true_values, predicted_values)),
            "mae": float(mean_absolute_error(true_values, predicted_values)),
        })
    return rows


def run_cross_validation(acoustic, text, targets, is_train) -> None:
    acoustic, text, targets = acoustic[is_train], text[is_train], targets[is_train]
    mlp_predictions = np.zeros_like(targets)
    ridge_predictions = np.zeros_like(targets)

    splitter = KFold(n_splits=base.NUMBER_OF_FOLDS, shuffle=True, random_state=base.RANDOM_SEED)
    for fold_index, (fit_index, held_out_index) in enumerate(splitter.split(targets)):
        print(f"Fold {fold_index + 1}")
        preprocessor = Preprocessor().fit(acoustic[fit_index], text[fit_index], targets[fit_index])
        model = train_model(preprocessor, acoustic[fit_index], text[fit_index], targets[fit_index],
                            seed=RANDOM_SEED + fold_index, verbose=(fold_index == 0))
        mlp_predictions[held_out_index] = predict(model, preprocessor, acoustic[held_out_index], text[held_out_index])

        ridge = base.build_ridge_model().fit(np.hstack([acoustic[fit_index], text[fit_index]]), targets[fit_index])
        ridge_predictions[held_out_index] = ridge.predict(np.hstack([acoustic[held_out_index], text[held_out_index]]))

    results = pd.DataFrame(score("fusion_mlp", targets, mlp_predictions) + score("ridge_all_features", targets, ridge_predictions))
    results.to_csv(base.OUTPUT_DIRECTORY / "fusion_mlp_cv_results.csv", index=False)
    print(results.round(3).to_string(index=False))


def run_test_evaluation(acoustic, text, targets, is_train) -> None:
    preprocessor = Preprocessor().fit(acoustic[is_train], text[is_train], targets[is_train])
    model = train_model(preprocessor, acoustic[is_train], text[is_train], targets[is_train],
                        seed=RANDOM_SEED, verbose=True)
    predictions = predict(model, preprocessor, acoustic[~is_train], text[~is_train])
    results = pd.DataFrame(score("fusion_mlp", targets[~is_train], predictions))
    results.to_csv(base.OUTPUT_DIRECTORY / "fusion_mlp_test_results.csv", index=False)
    print(results.round(3).to_string(index=False))

    model_directory = base.OUTPUT_DIRECTORY / "models"
    model_directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_directory / "fusion_mlp.pt")
    joblib.dump(preprocessor, model_directory / "fusion_mlp_preprocessor.joblib")


def main() -> None:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--evaluate_test", action="store_true")
    arguments = argument_parser.parse_args()

    acoustic, text, targets, is_train, _ = load_arrays()
    print(f"Train: {is_train.sum()}  Test: {(~is_train).sum()}")
    if arguments.evaluate_test:
        run_test_evaluation(acoustic, text, targets, is_train)
    else:
        run_cross_validation(acoustic, text, targets, is_train)


if __name__ == "__main__":
    main()