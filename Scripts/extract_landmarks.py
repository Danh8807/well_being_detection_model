import argparse
import pathlib
import subprocess
import tempfile
import traceback

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
LANDMARK_DIRECTORY = OUTPUT_DIRECTORY / "landmarks_mp_raw"
MODEL_PATH = OUTPUT_DIRECTORY / "models" / "face_landmarker.task"
ID_COLUMN = "ResponseId"
N_POINTS = 468


def extract_frames(video_path, frame_directory):
    """One frame per second, resized to 640 px wide (aspect ratio kept)."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vf", "fps=1,scale=640:-2", "-q:v", "2",
         str(frame_directory / "f_%05d.jpg")],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return sorted(frame_directory.glob("f_*.jpg"))


def landmarks_for_frame(image, landmarker):
    """936 values (x0, y0, x1, y1, ...) in pixels, or NaNs if no face is found."""
    height, width = image.shape[:2]
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not result.face_landmarks:
        return np.full(N_POINTS * 2, np.nan, dtype=np.float32)
    points = result.face_landmarks[0][:N_POINTS]
    return np.array([[p.x * width, p.y * height] for p in points], dtype=np.float32).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=pathlib.Path, default=OUTPUT_DIRECTORY / "dataset_all.csv")
    parser.add_argument("--maximum_samples", type=int, default=None)
    args = parser.parse_args()

    LANDMARK_DIRECTORY.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(args.dataset_path)
    if args.maximum_samples is not None:
        table = table.head(args.maximum_samples)

    options = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    summary, failed = [], []
    for _, row in tqdm(table.iterrows(), total=len(table)):
        identifier = row[ID_COLUMN]
        cache_path = LANDMARK_DIRECTORY / f"{identifier}.npy"
        if cache_path.exists():  # resume after interruption
            continue
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                frames = extract_frames(PROJECT_ROOT / row["video_path"], pathlib.Path(temporary_directory))
                sequence = np.stack([landmarks_for_frame(cv2.imread(str(f)), landmarker) for f in frames])
            np.save(cache_path, sequence)  # (seconds, 936), NaN where no face
            detected = int((~np.isnan(sequence[:, 0])).sum())
            summary.append((identifier, len(sequence), detected))
        except Exception:
            failed.append((identifier, traceback.format_exc()))

    summary_path = OUTPUT_DIRECTORY / "landmarks_mp_summary.csv"
    pd.DataFrame(summary, columns=[ID_COLUMN, "n_frames", "n_detected"]).to_csv(summary_path, index=False)
    pd.DataFrame(failed, columns=[ID_COLUMN, "error"]).to_csv(OUTPUT_DIRECTORY / "failed_landmarks_mp.csv", index=False)
    print(f"Done. Failed: {len(failed)}. Check {summary_path.name} for the face detection rate.")


if __name__ == "__main__":
    main()