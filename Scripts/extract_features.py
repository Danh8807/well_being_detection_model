import argparse
import json
import pathlib
import subprocess
import tempfile
import time
import traceback
import numpy as np
from scipy.io import wavfile
from typing import Optional

import opensmile
import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"
FEATURE_DIRECTORY = OUTPUT_DIRECTORY / "features_raw"
RESPONSE_IDENTIFIER_COLUMN = "ResponseId"


def parse_command_line_arguments() -> argparse.Namespace:
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--dataset_path", type=pathlib.Path, default=OUTPUT_DIRECTORY / "dataset_all.csv")
    argument_parser.add_argument("--maximum_samples", type=int, default=None)
    argument_parser.add_argument("--whisper_model_size", type=str, default="base")
    return argument_parser.parse_args()


def extract_audio_to_wav(video_path: pathlib.Path, wav_path: pathlib.Path) -> None:
    """Convert a video to 16 kHz mono wav, the input format of Whisper and openSMILE."""
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def build_acoustic_extractor() -> opensmile.Smile:
    """Functionals level gives one fixed-length vector (88 values) per file."""
    return opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )


def load_wav_as_float32(wav_path: pathlib.Path) -> np.ndarray:
    """
    Read a 16 kHz mono 16-bit wav file and return float32 samples in [-1, 1].
    Passing an array to Whisper avoids its internal decoder (PyAV).
    """
    sample_rate, samples = wavfile.read(str(wav_path))
    assert sample_rate == 16000, f"Unexpected sample rate: {sample_rate}"
    return samples.astype(np.float32) / 32768.0


def transcribe_audio(whisper_model: WhisperModel, wav_path: pathlib.Path) -> dict:
    """Return the transcript and simple speech statistics for one wav file."""
    audio_samples = load_wav_as_float32(wav_path)
    segment_generator, transcription_info = whisper_model.transcribe(
        audio_samples, language="en", vad_filter=True
    )
    # The generator can only be consumed once, so it is converted to a list.
    segment_list = list(segment_generator)
    transcript_text = " ".join(segment.text.strip() for segment in segment_list)
    return {
        "transcript": transcript_text,
        "word_count": len(transcript_text.split()),
        "speech_duration_seconds": sum(segment.end - segment.start for segment in segment_list),
        "audio_duration_seconds": transcription_info.duration,
    }


def process_single_video(
    participant_identifier: str,
    video_path: pathlib.Path,
    acoustic_extractor: opensmile.Smile,
    whisper_model: WhisperModel,
) -> Optional[float]:
    """
    Extract features for one video and write them to JSON.
    Returns the audio duration in seconds, or None if the file already existed.
    """
    output_path = FEATURE_DIRECTORY / f"{participant_identifier}.json"
    if output_path.exists():
        return None

    with tempfile.TemporaryDirectory() as temporary_directory:
        wav_path = pathlib.Path(temporary_directory) / "audio.wav"
        extract_audio_to_wav(video_path, wav_path)
        acoustic_table = acoustic_extractor.process_file(str(wav_path))
        acoustic_features = acoustic_table.iloc[0].to_dict()
        text_statistics = transcribe_audio(whisper_model, wav_path)

    record = {
        "participant_identifier": participant_identifier,
        "acoustic_features": acoustic_features,
        **text_statistics,
    }
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(record, output_file)
    return text_statistics["audio_duration_seconds"]


def main() -> None:
    arguments = parse_command_line_arguments()
    FEATURE_DIRECTORY.mkdir(parents=True, exist_ok=True)

    dataset_table = pd.read_csv(arguments.dataset_path)
    if arguments.maximum_samples is not None:
        dataset_table = dataset_table.head(arguments.maximum_samples)

    whisper_model = WhisperModel(arguments.whisper_model_size, device="cpu", compute_type="int8")
    acoustic_extractor = build_acoustic_extractor()

    failed_records = []
    processed_audio_seconds = 0.0
    start_time = time.perf_counter()

    for _, row in tqdm(dataset_table.iterrows(), total=len(dataset_table)):
        try:
            audio_duration = process_single_video(
                participant_identifier=row[RESPONSE_IDENTIFIER_COLUMN],
                video_path=PROJECT_ROOT / row["video_path"],
                acoustic_extractor=acoustic_extractor,
                whisper_model=whisper_model,
            )
            if audio_duration is not None:
                processed_audio_seconds += audio_duration
        except Exception:
            # A bad clip is logged and skipped so the long job keeps running.
            failed_records.append((row[RESPONSE_IDENTIFIER_COLUMN], traceback.format_exc()))

    elapsed_seconds = time.perf_counter() - start_time
    if processed_audio_seconds > 0:
        print(
            f"Processed {processed_audio_seconds / 60:.1f} audio minutes in "
            f"{elapsed_seconds / 60:.1f} minutes "
            f"(real-time factor {processed_audio_seconds / elapsed_seconds:.2f})."
        )
    print(f"Failed clips: {len(failed_records)}")
    pd.DataFrame(failed_records, columns=[RESPONSE_IDENTIFIER_COLUMN, "error"]).to_csv(
        OUTPUT_DIRECTORY / "failed_clips.csv", index=False
    )


if __name__ == "__main__":
    main()