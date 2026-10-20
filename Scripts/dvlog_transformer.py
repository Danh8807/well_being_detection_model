import argparse
import copy
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import baseline_ridge as base

D_MODEL = 32
N_HEADS = 4
FEED_FORWARD = 64
N_LAYERS = 1
DROPOUT = 0.3
LEARNING_RATE = 0.0002
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
PATIENCE = 10
LABELS = ["SWLS", "PA", "NA"]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return x + self.encoding[: x.size(1)]


def downsampled_padding(valid):
    """Padding mask after two stride-2 convolutions (length L -> ceil(L / 4))."""
    lengths = valid.sum(1)
    new_lengths = ((lengths + 1) // 2 + 1) // 2
    new_total = ((valid.size(1) + 1) // 2 + 1) // 2
    return torch.arange(new_total, device=valid.device)[None, :] >= new_lengths[:, None]


class UnimodalEncoder(nn.Module):
    """Downsample by 4 with 1-D convolutions, add positions, then a Transformer encoder."""

    def __init__(self, d_in):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(d_in, D_MODEL, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv1d(D_MODEL, D_MODEL, 3, stride=2, padding=1), nn.ReLU(),
        )
        self.position = PositionalEncoding(D_MODEL)
        layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, FEED_FORWARD, DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)

    def forward(self, x, pad_mask):
        hidden = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.encoder(self.position(hidden), src_key_padding_mask=pad_mask)


class CrossFusion(nn.Module):
    """A->V and V->A cross-attention, residual + norm, concatenate, then a Transformer encoder."""

    def __init__(self):
        super().__init__()
        self.audio_to_face = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.face_to_audio = nn.MultiheadAttention(D_MODEL, N_HEADS, dropout=DROPOUT, batch_first=True)
        self.norm_audio = nn.LayerNorm(D_MODEL)
        self.norm_face = nn.LayerNorm(D_MODEL)
        layer = nn.TransformerEncoderLayer(2 * D_MODEL, N_HEADS, 2 * FEED_FORWARD, DROPOUT, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS, enable_nested_tensor=False)

    def forward(self, audio, face, pad_mask):
        cross_audio, _ = self.audio_to_face(audio, face, face, key_padding_mask=pad_mask)
        cross_face, _ = self.face_to_audio(face, audio, audio, key_padding_mask=pad_mask)
        joined = torch.cat([self.norm_audio(audio + cross_audio), self.norm_face(face + cross_face)], dim=-1)
        return self.encoder(joined, src_key_padding_mask=pad_mask)


class DVlogRegressor(nn.Module):
    def __init__(self, d_audio, d_face, modality):
        super().__init__()
        self.modality = modality
        if modality in ("both", "audio"):
            self.audio_encoder = UnimodalEncoder(d_audio)
        if modality in ("both", "face"):
            self.face_encoder = UnimodalEncoder(d_face)
        if modality == "both":
            self.fusion = CrossFusion()
        width = 2 * D_MODEL if modality == "both" else D_MODEL
        self.dropout = nn.Dropout(DROPOUT)
        self.head = nn.Linear(width, 3)

    def forward(self, audio, face, valid):
        pad_mask = downsampled_padding(valid)
        if self.modality == "audio":
            z = self.audio_encoder(audio, pad_mask)
        elif self.modality == "face":
            z = self.face_encoder(face, pad_mask)
        else:
            z = self.fusion(self.audio_encoder(audio, pad_mask), self.face_encoder(face, pad_mask), pad_mask)
        keep = (~pad_mask).unsqueeze(-1).float()
        pooled = (z * keep).sum(1) / keep.sum(1)       # global average pooling over real steps
        return self.head(self.dropout(pooled))


def train_and_predict(tensors, y, fit_index, new_index, modality, seed, epochs):
    audio, face, valid = tensors
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(fit_index)
    n_validation = max(1, int(0.1 * len(order)))
    validation_index, train_index = order[:n_validation], order[n_validation:]

    mean, std = y[train_index].mean(axis=0), y[train_index].std(axis=0)
    target = torch.tensor((y - mean) / std, dtype=torch.float32)

    model = DVlogRegressor(audio.shape[2], face.shape[2], modality)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_function = nn.MSELoss()
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
            outputs.append(model(audio[idx], face[idx], valid[idx]).numpy())
    return np.concatenate(outputs) * std + mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["both", "audio", "face"], default="both")
    parser.add_argument("--n_seeds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    data = np.load(base.OUTPUT_DIRECTORY / "sequences.npz", allow_pickle=True)
    is_train = data["split"] == "train"      # the test split is never touched here
    tensors = (
        torch.tensor(data["audio"][is_train]),
        torch.tensor(data["face"][is_train]),
        torch.tensor(data["valid"][is_train]),
    )
    y = data["targets"][is_train].astype(float)
    print("Train participants:", len(y), "| modality:", args.modality)

    folds = list(KFold(5, shuffle=True, random_state=base.RANDOM_SEED).split(y))
    rows = []
    for seed_number in range(args.n_seeds):
        predicted = np.zeros_like(y)
        for k, (fit_index, held_index) in enumerate(folds):
            predicted[held_index] = train_and_predict(
                tensors, y, fit_index, held_index, args.modality,
                seed=base.RANDOM_SEED + 10 * seed_number + k, epochs=args.epochs)
            print(f"  seed {seed_number} fold {k} done", flush=True)
        for i, label in enumerate(LABELS):
            lower, upper = base.bootstrap_interval_for_pearson(y[:, i], predicted[:, i])
            rows.append({"seed": seed_number, "target": label,
                         "pearson_r": base.pearson_or_zero(y[:, i], predicted[:, i]),
                         "ci_lower": lower, "ci_upper": upper})
    results = pd.DataFrame(rows)
    out_path = base.OUTPUT_DIRECTORY / f"dvlog_transformer_cv_{args.modality}.csv"
    results.to_csv(out_path, index=False)
    print(results.round(3).to_string(index=False))
    print("Mean r across seeds:", results.groupby("target")["pearson_r"].mean().round(3).to_dict())
    print("Saved", out_path)


if __name__ == "__main__":
    main()