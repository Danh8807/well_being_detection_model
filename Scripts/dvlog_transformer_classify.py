import argparse
import copy
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base
from dvlog_transformer import DVlogRegressor, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, PATIENCE

LOW_MAX, MID_MAX = 10, 20      # same fixed classes: <=10 | 11-20 | >=21


def train_and_predict(tensors, y, fit_index, new_index, modality, seed, epochs):
    audio, face, valid = tensors
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(fit_index)
    n_validation = max(1, int(0.1 * len(order)))
    validation_index, train_index = order[:n_validation], order[n_validation:]

    counts = np.bincount(y[train_index], minlength=3)
    weights = torch.tensor(len(train_index) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    target = torch.tensor(y, dtype=torch.long)

    model = DVlogRegressor(audio.shape[2], face.shape[2], modality)   # head gives 3 logits
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.CrossEntropyLoss(weight=weights)               # balances the classes
    best_loss, best_state, wait = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        shuffled = rng.permutation(train_index)
        for start in range(0, len(shuffled), BATCH_SIZE):
            idx = shuffled[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            loss = loss_function(model(audio[idx], face[idx], valid[idx]), target[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(
                model(audio[validation_index], face[validation_index], valid[validation_index]),
                target[validation_index]).item()
        if validation_loss < best_loss:
            best_loss, best_state, wait = validation_loss, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(new_index), 128):
            idx = new_index[start:start + 128]
            outputs.append(torch.softmax(model(audio[idx], face[idx], valid[idx]), dim=1).numpy())
    return np.concatenate(outputs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["both", "audio", "face"], default="both")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    data = np.load(base.OUTPUT_DIRECTORY / "sequences.npz", allow_pickle=True)
    is_train = data["split"] == "train"      # the test split is never touched here
    tensors = (torch.tensor(data["audio"][is_train]), torch.tensor(data["face"][is_train]),
               torch.tensor(data["valid"][is_train]))
    swls = data["targets"][is_train][:, 0]
    y = np.where(swls <= LOW_MAX, 0, np.where(swls <= MID_MAX, 1, 2))
    counts = np.bincount(y, minlength=3)
    print("Train:", len(y), "| class counts (low/middle/high):", counts.tolist(),
          "| majority-class accuracy %.3f (not a skill)" % (counts.max() / len(y)))

    rows = []
    for seed_number in range(args.n_seeds):
        proba = np.zeros((len(y), 3))
        splitter = StratifiedKFold(5, shuffle=True, random_state=base.RANDOM_SEED)
        for k, (fit_index, held_index) in enumerate(splitter.split(np.zeros(len(y)), y)):
            proba[held_index] = train_and_predict(tensors, y, fit_index, held_index, args.modality,
                                                  seed=base.RANDOM_SEED + 10 * seed_number + k,
                                                  epochs=args.epochs)
            print(f"  seed {seed_number} fold {k} done", flush=True)
        predicted = proba.argmax(axis=1)
        rows.append({"seed": seed_number,
                     "balanced_acc": balanced_accuracy_score(y, predicted),
                     "macro_auc": roc_auc_score(y, proba, multi_class="ovr", average="macro"),
                     "macro_f1": f1_score(y, predicted, average="macro")})
    results = pd.DataFrame(rows)
    out_path = base.OUTPUT_DIRECTORY / f"dvlog_transformer_classify_{args.modality}.csv"
    results.to_csv(out_path, index=False)
    print("\nChance: balanced accuracy 0.333, macro AUC 0.500")
    print(results.round(3).to_string(index=False))
    print("Saved", out_path)


if __name__ == "__main__":
    main()