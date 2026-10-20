# Scripts/extract_audio_embeddings.py
import argparse
import pathlib
import subprocess
import tempfile
import traceback

import numpy as np
import pandas as pd
import torch
from scipy.io import wavfile
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
ID_COLUMN = "ResponseId"
CHUNK_SAMPLES = 20 * 16000
MIN_SAMPLES = 4000


def extract_audio_to_wav(video_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def embed_wave(wave, extractor, model):
    """Mean over time of every layer's hidden states -> (n_layers, dim)."""
    total, frames = None, 0
    for start in range(0, len(wave), CHUNK_SAMPLES):
        piece = wave[start:start + CHUNK_SAMPLES]
        if len(piece) < MIN_SAMPLES:
            continue
        inputs = extractor(piece, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True).hidden_states
        stack = torch.stack(hidden)[:, 0]  # (layers, frames, dim)
        summed = stack.sum(1).numpy()
        total = summed if total is None else total + summed
        frames += stack.shape[1]
    return (total / frames).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    parser.add_argument("--dataset_path", type=pathlib.Path, default=OUTPUT_DIRECTORY / "dataset_all.csv")
    parser.add_argument("--maximum_samples", type=int, default=None)
    args = parser.parse_args()

    name = args.model.split("/")[-1]
    cache_directory = OUTPUT_DIRECTORY / f"audio_emb_raw_{name}"
    cache_directory.mkdir(parents=True, exist_ok=True)

    table = pd.read_csv(args.dataset_path)
    if args.maximum_samples is not None:
        table = table.head(args.maximum_samples)

    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval()

    failed = []
    for _, row in tqdm(table.iterrows(), total=len(table)):
        identifier = row[ID_COLUMN]
        cache_path = cache_directory / f"{identifier}.npy"
        if cache_path.exists():  # resume after interruption
            continue
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                wav_path = pathlib.Path(temporary_directory) / "audio.wav"
                extract_audio_to_wav(PROJECT_ROOT / row["video_path"], wav_path)
                rate, samples = wavfile.read(str(wav_path))
            wave = samples.astype(np.float32) / 32768.0
            np.save(cache_path, embed_wave(wave, extractor, model))
        except Exception:
            failed.append((identifier, traceback.format_exc()))

    identifiers, embeddings = [], []
    for path in sorted(cache_directory.glob("*.npy")):
        identifiers.append(path.stem)
        embeddings.append(np.load(path))
    out_path = OUTPUT_DIRECTORY / f"audio_emb_{name}.npz"
    np.savez(out_path, ids=np.array(identifiers), embeddings=np.stack(embeddings))
    print(f"Saved {out_path} ({len(identifiers)} people). Failed: {len(failed)}")
    pd.DataFrame(failed, columns=[ID_COLUMN, "error"]).to_csv(
        OUTPUT_DIRECTORY / f"failed_audio_emb_{name}.csv", index=False
    )


if __name__ == "__main__":
    main()