import argparse
import pathlib
import subprocess
import tempfile
import traceback

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
LANDMARK_DIRECTORY = OUTPUT_DIRECTORY / "landmarks_mp_raw"
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


def landmarks_for_frame(image, face_mesh):
    """936 values (x0, y0, x1, y1, ...) in pixels, or NaNs if no face is found."""
    height, width = image.shape[:2]
    result = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not result.multi_face_landmarks:
        return np.full(N_POINTS * 2, np.nan, dtype=np.float32)
    points = result.multi_face_landmarks[0].landmark[:N_POINTS]
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

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, refine_landmarks=False, min_detection_confidence=0.5
    )

    summary, failed = [], []
    for _, row in tqdm(table.iterrows(), total=len(table)):
        identifier = row[ID_COLUMN]
        cache_path = LANDMARK_DIRECTORY / f"{identifier}.npy"
        if cache_path.exists():  # resume after interruption
            continue
        try:
            with tempfile.TemporaryDirectory() as temporary_directory:
                frames = extract_frames(PROJECT_ROOT / row["video_path"], pathlib.Path(temporary_directory))
                sequence = np.stack([landmarks_for_frame(cv2.imread(str(f)), face_mesh) for f in frames])
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