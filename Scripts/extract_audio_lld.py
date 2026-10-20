import argparse
import pathlib
import subprocess
import tempfile
import traceback

import numpy as np
import opensmile
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
LLD_DIRECTORY = OUTPUT_DIRECTORY / "audio_lld_raw"
ID_COLUMN = "ResponseId"


def extract_audio_to_wav(video_path, wav_path):
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def per_second_means(smile, wav_path):
    """Run openSMILE once, then average the 10 ms frames inside each 1 s window."""
    frames = smile.process_file(str(wav_path))
    seconds = np.floor(frames.index.get_level_values("start").total_seconds()).astype(int)
    grouped = frames.groupby(seconds).mean()  # NaNs (e.g. unvoiced pitch) are skipped
    grouped = grouped.reindex(range(int(seconds.max()) + 1))
    return grouped.to_numpy(dtype=np.float32), list(frames.columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=pathlib.Path, default=OUTPUT_DIRECTORY / "dataset_all.csv")
    parser.add_argument("--maximum_samples", type=int, default=None)
    args = parser.parse_args()

    LLD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.dataset_path)
    if args.maximum_samples is not None:
        table = table.head(args.maximum_samples)

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    summary, failed, column_names = [], [], None
    for _, row in tqdm(table.iterrows(), total=len(table)):
        identifier = row[ID_COLUMN]
        cache_path = LLD_DIRECTORY / f"{identifier}.npy"
        if cache_path.exists():  # resume after interruption
            continue
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                wav_path = pathlib.Path(temporary_directory) / "audio.wav"
                extract_audio_to_wav(PROJECT_ROOT / row["video_path"], wav_path)
                sequence, column_names = per_second_means(smile, wav_path)
            np.save(cache_path, sequence)  # (seconds, n_descriptors)
            summary.append((identifier, sequence.shape[0], sequence.shape[1]))
        except Exception:
            failed.append((identifier, traceback.format_exc()))

    if column_names is not None:
        pd.Series(column_names).to_csv(OUTPUT_DIRECTORY / "audio_lld_columns.csv", index=False, header=False)
    pd.DataFrame(summary, columns=[ID_COLUMN, "n_seconds", "n_descriptors"]).to_csv(
        OUTPUT_DIRECTORY / "audio_lld_summary.csv", index=False)
    pd.DataFrame(failed, columns=[ID_COLUMN, "error"]).to_csv(
        OUTPUT_DIRECTORY / "failed_audio_lld.csv", index=False)
    print(f"Done. Failed: {len(failed)}. Descriptors per second: "
          f"{len(column_names) if column_names else 'n/a'}")


if __name__ == "__main__":
    main()