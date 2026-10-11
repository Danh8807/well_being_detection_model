import collections
import pathlib
from typing import Dict, List, Tuple

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "Data"

MEDIA_EXTENSION = ".mp4"
SURVEY_EXTENSION = ".csv"

SCORE_ITEM_COLUMNS: List[str] = (
    [f"SWLS_{index}" for index in range(1, 6)]
    + [f"PA_{index}" for index in range(1, 7)]
    + [f"NA_{index}" for index in range(1, 7)]
)

# Video codes shorter than this are ignored, to avoid accidental substring matches.
MINIMUM_VIDEO_CODE_LENGTH = 20
NUMBER_OF_EXAMPLES_TO_PRINT = 5


def find_files_with_extension(data_directory: pathlib.Path, extension: str) -> List[pathlib.Path]:
    """Return all files under the directory with the given extension (case-insensitive)."""
    return [
        file_path
        for file_path in data_directory.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() == extension
    ]


def report_media_files(media_paths: List[pathlib.Path]) -> None:
    """Print the number of media files, their sizes, and a few example names."""
    print(f"Media files ({MEDIA_EXTENSION}) found: {len(media_paths)}")
    if not media_paths:
        return
    total_bytes = sum(path.stat().st_size for path in media_paths)
    print(f"Total size:   {total_bytes / 1e9:.2f} GB")
    print(f"Average size: {total_bytes / len(media_paths) / 1e6:.2f} MB per file")
    print("Example file names:")
    for media_path in media_paths[:NUMBER_OF_EXAMPLES_TO_PRINT]:
        print(f"  {media_path.name}")


def describe_finished_and_scores(survey_table: pd.DataFrame) -> None:
    """
    Print the values of Finished and, for every score item column, each
    distinct text value with its count. Item values are stored as text such
    as "4=quite a bit" or "Strongly agree", so the exact labels are needed
    to build the numeric conversion.
    """
    print(f"\nSurvey shape: {survey_table.shape[0]} rows, {survey_table.shape[1]} columns")
    print("\nValue counts of Finished (including missing):")
    print(survey_table["Finished"].value_counts(dropna=False).to_string())

    print("\nDistinct values per score item column:")
    for column_name in SCORE_ITEM_COLUMNS:
        if column_name not in survey_table.columns:
            print(f"\n  {column_name}: COLUMN NOT FOUND")
            continue
        print(f"\n  {column_name}:")
        print(survey_table[column_name].value_counts(dropna=False).to_string())


def collect_video_code_lookup(survey_table: pd.DataFrame) -> Tuple[Dict[str, List[Tuple[int, str]]], List[str]]:
    """
    Build a dictionary from each video code to the list of (row index, column
    name) cells that contain it. Only columns whose name contains "video"
    are used. Also prints how many cells are filled in each such column.
    """
    video_columns = [c for c in survey_table.columns if "video" in c.lower()]
    code_lookup: Dict[str, List[Tuple[int, str]]] = collections.defaultdict(list)

    print(f"\nColumns containing 'video': {len(video_columns)}")
    for column_name in video_columns:
        filled_cells = survey_table[column_name].dropna()
        filled_cells = filled_cells[filled_cells.str.len() >= MINIMUM_VIDEO_CODE_LENGTH]
        print(f"  {column_name}: {len(filled_cells)} filled cells")
        for row_index, cell_value in filled_cells.items():
            code_lookup[cell_value.strip()].append((row_index, column_name))
    return code_lookup, video_columns


def find_code_in_file_name(file_stem: str, code_lookup: Dict[str, List[Tuple[int, str]]]) -> List[str]:
    """
    Return every video code that appears inside the file name. A file name
    normally contains at most one code; more than one means the rule is ambiguous.
    """
    return [code for code in code_lookup if code in file_stem]


def trial_match_by_video_code(
    survey_table: pd.DataFrame,
    media_paths: List[pathlib.Path],
    code_lookup: Dict[str, List[Tuple[int, str]]],
) -> None:
    """
    For each media file, find the survey cell whose code is contained in the
    file name. Report counts of matched, unmatched and ambiguous files, which
    column the matches fall in, and the Finished values of the matched rows.
    """
    unmatched_count = 0
    ambiguous_count = 0
    matched_column_counter: collections.Counter = collections.Counter()
    matched_finished_counter: collections.Counter = collections.Counter()
    matched_row_indices: List[int] = []
    example_pairs: List[Tuple[str, str]] = []

    for media_path in media_paths:
        matching_codes = find_code_in_file_name(media_path.stem, code_lookup)
        if not matching_codes:
            unmatched_count += 1
            continue
        if len(matching_codes) > 1:
            ambiguous_count += 1
            continue
        code = matching_codes[0]
        cells = code_lookup[code]
        for row_index, column_name in cells:
            matched_column_counter[column_name] += 1
            matched_finished_counter[survey_table.loc[row_index, "Finished"]] += 1
            matched_row_indices.append(row_index)
        if len(example_pairs) < NUMBER_OF_EXAMPLES_TO_PRINT:
            example_pairs.append((media_path.name, code))

    print("\nTrial match by video code (file name contains the code):")
    print(f"  Files matched to a code:     {len(media_paths) - unmatched_count - ambiguous_count}")
    print(f"  Files with no matching code: {unmatched_count}")
    print(f"  Files with several codes:    {ambiguous_count}")
    print(f"  Rows matched more than once: {len(matched_row_indices) - len(set(matched_row_indices))}")
    print("  Matched cells per column:")
    for column_name, count in matched_column_counter.most_common():
        print(f"    {column_name}: {count}")
    print("  Finished values of matched rows:")
    for finished_value, count in matched_finished_counter.most_common():
        print(f"    {finished_value}: {count}")
    print("  Example (file name, code found in it):")
    for file_name, code in example_pairs:
        print(f"    {file_name}  ->  {code}")


def trial_match_by_response_identifier(
    survey_table: pd.DataFrame, media_paths: List[pathlib.Path]
) -> None:
    """
    Alternative rule: check whether file names contain the ResponseId
    (the "R_..." value). Reports how many files contain one.
    """
    response_identifiers = set(survey_table["ResponseId"].dropna().str.strip())
    files_with_identifier = sum(
        any(identifier in media_path.stem for identifier in response_identifiers)
        for media_path in media_paths
    )
    print(f"\nFiles whose name contains a ResponseId: {files_with_identifier} of {len(media_paths)}")


def main() -> None:
    media_paths = find_files_with_extension(DATA_DIRECTORY, MEDIA_EXTENSION)
    report_media_files(media_paths)

    survey_paths = find_files_with_extension(DATA_DIRECTORY, SURVEY_EXTENSION)
    print("\nSurvey CSV candidates:")
    for survey_path in survey_paths:
        print(f"  {survey_path.relative_to(PROJECT_ROOT)} ({survey_path.stat().st_size / 1e6:.1f} MB)")
    if not survey_paths:
        print("No CSV found under Data/.")
        return

    # If several CSV files exist, the first one is inspected; tell me which is correct.
    survey_table = pd.read_csv(survey_paths[0], dtype=str, low_memory=False)
    print(f"\nInspecting survey: {survey_paths[0].name}")

    describe_finished_and_scores(survey_table)
    code_lookup, _ = collect_video_code_lookup(survey_table)
    trial_match_by_video_code(survey_table, media_paths, code_lookup)
    trial_match_by_response_identifier(survey_table, media_paths)


if __name__ == "__main__":
    main() 
