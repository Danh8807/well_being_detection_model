"""
preprocess_data.py

Step 1 of the pipeline: build the labelled dataset.

For every mp4 file on disk:
  1. Find the survey row whose video code (columns question7_video or
     question9s_video) equals the file name without extension.
  2. Keep the row only if Finished is TRUE.
  3. Convert the SWLS and PANAS item answers from text to numbers and
     compute three scores: swls_score, positive_affect_score,
     negative_affect_score.
  4. Drop rows with any missing item.
  5. Split into 80 percent train and 20 percent test with a fixed seed.

Outputs (all written to outputs/, which is excluded from git):
  dataset_all.csv, train.csv, test.csv

Example usage (from the project root):
    python Scripts/preprocess_data.py
"""

import pathlib
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "Data"
OUTPUT_DIRECTORY = PROJECT_ROOT / "outputs"

SURVEY_FILE_NAME = "Connect_AVI_VIA_WB.csv"
MEDIA_EXTENSION = ".mp4"

# The only columns that contain the video code of the recorded answer.
VIDEO_COLUMNS_TO_USE: List[str] = ["question7_video", "question9s_video"]
RESPONSE_IDENTIFIER_COLUMN = "ResponseId"

SWLS_ITEM_COLUMNS: List[str] = [f"SWLS_{index}" for index in range(1, 6)]
POSITIVE_AFFECT_ITEM_COLUMNS: List[str] = [f"PA_{index}" for index in range(1, 7)]
NEGATIVE_AFFECT_ITEM_COLUMNS: List[str] = [f"NA_{index}" for index in range(1, 7)]

# This survey uses a 5-point answer format for SWLS (not the 7-point original).
SWLS_TEXT_TO_NUMBER: Dict[str, int] = {
    "Strongly disagree": 1,
    "Disagree": 2,
    "Neither agree nor disagree": 3,
    "Agree": 4,
    "Strongly agree": 5,
}

# Kept only for later fairness analysis; never used as model input.
DEMOGRAPHIC_COLUMNS: List[str] = ["Age", "Gender", "Race", "Device type"]

TEST_FRACTION = 0.2
RANDOM_SEED = 42


def convert_swls_items_to_numbers(survey_table: pd.DataFrame) -> pd.DataFrame:
    """
    Replace the text answers of the SWLS items by numbers 1 to 5.
    Raises an error if an unexpected label appears, so that nothing is
    silently turned into a missing value.
    """
    converted_table = survey_table.copy()
    for column_name in SWLS_ITEM_COLUMNS:
        stripped_values = converted_table[column_name].str.strip()
        numeric_values = stripped_values.map(SWLS_TEXT_TO_NUMBER)
        unexpected_labels = stripped_values[stripped_values.notna() & numeric_values.isna()].unique()
        if len(unexpected_labels) > 0:
            raise ValueError(f"Unexpected labels in {column_name}: {list(unexpected_labels)}")
        converted_table[column_name] = numeric_values
    return converted_table


def convert_affect_items_to_numbers(survey_table: pd.DataFrame) -> pd.DataFrame:
    """
    Convert PANAS answers such as "4=quite a bit" to the number 4 by
    reading the leading digit. Raises an error if a non-empty answer
    does not start with a digit from 1 to 5.
    """
    converted_table = survey_table.copy()
    for column_name in POSITIVE_AFFECT_ITEM_COLUMNS + NEGATIVE_AFFECT_ITEM_COLUMNS:
        leading_digit = converted_table[column_name].str.extract(r"^\s*([1-5])\s*=")[0]
        has_answer = converted_table[column_name].notna()
        if (has_answer & leading_digit.isna()).any():
            bad_values = converted_table.loc[has_answer & leading_digit.isna(), column_name].unique()
            raise ValueError(f"Unexpected answers in {column_name}: {list(bad_values)}")
        converted_table[column_name] = leading_digit.astype(float)
    return converted_table


def compute_target_scores(survey_table: pd.DataFrame) -> pd.DataFrame:
    """
    Add the three target scores as sums of their items.
    skipna=False means a single missing item makes the whole score missing,
    which is safer than silently summing fewer items.
    """
    scored_table = survey_table.copy()
    scored_table["swls_score"] = scored_table[SWLS_ITEM_COLUMNS].sum(axis=1, skipna=False)
    scored_table["positive_affect_score"] = scored_table[POSITIVE_AFFECT_ITEM_COLUMNS].sum(
        axis=1, skipna=False
    )
    scored_table["negative_affect_score"] = scored_table[NEGATIVE_AFFECT_ITEM_COLUMNS].sum(
        axis=1, skipna=False
    )
    return scored_table


def build_video_code_table(survey_table: pd.DataFrame) -> pd.DataFrame:
    """
    Return a table with one row per filled video cell in the chosen columns:
    video_code, ResponseId, and the column in which the code was found.
    """
    pieces = []
    for column_name in VIDEO_COLUMNS_TO_USE:
        filled_rows = survey_table[survey_table[column_name].notna()]
        pieces.append(
            pd.DataFrame(
                {
                    "video_code": filled_rows[column_name].str.strip(),
                    RESPONSE_IDENTIFIER_COLUMN: filled_rows[RESPONSE_IDENTIFIER_COLUMN],
                    "source_column": column_name,
                }
            )
        )
    code_table = pd.concat(pieces, ignore_index=True)

    duplicated_code_count = int(code_table["video_code"].duplicated().sum())
    print(f"Video codes in the chosen columns: {len(code_table)} "
          f"(duplicated codes: {duplicated_code_count})")
    return code_table.drop_duplicates(subset=["video_code"], keep=False)


def build_media_table(data_directory: pathlib.Path) -> pd.DataFrame:
    """List every mp4 file on disk with its file name stem (the video code)."""
    media_paths = sorted(
        path for path in data_directory.rglob("*")
        if path.is_file() and path.suffix.lower() == MEDIA_EXTENSION
    )
    return pd.DataFrame(
        {
            "video_code": [path.stem for path in media_paths],
            "video_path": [path.relative_to(PROJECT_ROOT).as_posix() for path in media_paths],
        }
    )


def explain_unmatched_files(unmatched_codes: List[str], survey_table: pd.DataFrame) -> None:
    """
    For every media file without a match, search all columns containing
    "video" to see whether its code appears elsewhere in the survey.
    """
    video_columns = [c for c in survey_table.columns if "video" in c.lower()]
    print("Unmatched files and where their code appears in the survey:")
    for video_code in unmatched_codes:
        found_columns = [
            column_name for column_name in video_columns
            if (survey_table[column_name].str.strip() == video_code).any()
        ]
        print(f"  {video_code}: {found_columns if found_columns else 'not found anywhere'}")


def build_dataset(survey_table: pd.DataFrame, media_table: pd.DataFrame) -> pd.DataFrame:
    """
    Run the whole filtering funnel and print the number of participants
    remaining after each step.
    """
    code_table = build_video_code_table(survey_table)

    matched_table = media_table.merge(code_table, on="video_code", how="left")
    unmatched_codes = matched_table.loc[
        matched_table[RESPONSE_IDENTIFIER_COLUMN].isna(), "video_code"
    ].tolist()
    print(f"\nMedia files on disk:              {len(media_table)}")
    print(f"Files matched to a survey row:    {len(media_table) - len(unmatched_codes)}")
    if unmatched_codes:
        explain_unmatched_files(unmatched_codes, survey_table)
    matched_table = matched_table.dropna(subset=[RESPONSE_IDENTIFIER_COLUMN])

    # One survey row must correspond to one video, otherwise a person would
    # appear twice and could leak between the train and test sets.
    repeated_count = int(matched_table[RESPONSE_IDENTIFIER_COLUMN].duplicated().sum())
    print(f"Participants with more than one video: {repeated_count} (kept only the first)")
    matched_table = matched_table.drop_duplicates(subset=[RESPONSE_IDENTIFIER_COLUMN])

    survey_columns_to_keep = (
        [RESPONSE_IDENTIFIER_COLUMN, "Finished"]
        + SWLS_ITEM_COLUMNS + POSITIVE_AFFECT_ITEM_COLUMNS + NEGATIVE_AFFECT_ITEM_COLUMNS
        + [c for c in DEMOGRAPHIC_COLUMNS if c in survey_table.columns]
    )
    dataset = matched_table.merge(
        survey_table[survey_columns_to_keep].drop_duplicates(subset=[RESPONSE_IDENTIFIER_COLUMN]),
        on=RESPONSE_IDENTIFIER_COLUMN,
        how="inner",
    )
    print(f"After joining with survey rows:   {len(dataset)}")

    finished_mask = dataset["Finished"].str.strip().str.upper() == "TRUE"
    print(f"Removed because Finished is not TRUE: {int((~finished_mask).sum())}")
    dataset = dataset[finished_mask]

    dataset = compute_target_scores(convert_affect_items_to_numbers(convert_swls_items_to_numbers(dataset)))
    target_columns = ["swls_score", "positive_affect_score", "negative_affect_score"]
    missing_mask = dataset[target_columns].isna().any(axis=1)
    print(f"Removed because a score item is missing: {int(missing_mask.sum())}")
    dataset = dataset[~missing_mask]

    print(f"Final number of participants:     {len(dataset)}")
    return dataset.reset_index(drop=True)


def split_train_test(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Add a "split" column with the values "train" and "test" (80 / 20).
    Each participant has exactly one video, so a random split by row is
    already a split by participant and cannot leak between the two sets.
    """
    assert dataset[RESPONSE_IDENTIFIER_COLUMN].is_unique
    train_indices, test_indices = train_test_split(
        dataset.index, test_size=TEST_FRACTION, random_state=RANDOM_SEED, shuffle=True
    )
    split_dataset = dataset.copy()
    split_dataset["split"] = "train"
    split_dataset.loc[test_indices, "split"] = "test"
    return split_dataset


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    survey_table = pd.read_csv(DATA_DIRECTORY / SURVEY_FILE_NAME, dtype=str, low_memory=False)
    media_table = build_media_table(DATA_DIRECTORY)

    dataset = split_train_test(build_dataset(survey_table, media_table))

    output_columns = (
        [RESPONSE_IDENTIFIER_COLUMN, "video_code", "video_path", "source_column", "split",
         "swls_score", "positive_affect_score", "negative_affect_score"]
        + [c for c in DEMOGRAPHIC_COLUMNS if c in dataset.columns]
    )
    dataset[output_columns].to_csv(OUTPUT_DIRECTORY / "dataset_all.csv", index=False)
    dataset[dataset["split"] == "train"][output_columns].to_csv(OUTPUT_DIRECTORY / "train.csv", index=False)
    dataset[dataset["split"] == "test"][output_columns].to_csv(OUTPUT_DIRECTORY / "test.csv", index=False)

    target_columns = ["swls_score", "positive_affect_score", "negative_affect_score"]
    print("\nSplit sizes:")
    print(dataset["split"].value_counts().to_string())
    print("\nTarget score summary per split (train and test should look similar):")
    print(dataset.groupby("split")[target_columns].agg(["mean", "std", "min", "max"]).round(2).T.to_string())
    print("\nCorrelation between the three targets:")
    print(dataset[target_columns].corr().round(2).to_string())
    print("\nVideos per source column:")
    print(dataset["source_column"].value_counts().to_string())


if __name__ == "__main__":
    main()