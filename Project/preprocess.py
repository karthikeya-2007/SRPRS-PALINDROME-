
from __future__ import annotations

import html
import re
from pathlib import Path

import pandas as pd


INPUT_FILE = "unified_papers.csv"
OUTPUT_FILE = "processed_papers.csv"


# ---------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------

def clean_text(text: object) -> str:
    """
    Clean scientific text while preserving useful semantic information.

    Operations:
    - HTML entity decoding
    - LaTeX command removal
    - LaTeX math delimiters removal
    - citation cleanup
    - whitespace normalization
    - removal of unusual symbols
    """

    if text is None or pd.isna(text):
        return ""

    text = str(text)

    # Decode HTML entities such as &amp;
    text = html.unescape(text)

    # Remove common LaTeX math delimiters.
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$.*?\$", " ", text)

    # Remove common LaTeX commands:
    # \frac, \textbf, \emph, etc.
    text = re.sub(
        r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?\{[^{}]*\}",
        " ",
        text,
    )

    # Remove remaining LaTeX commands.
    text = re.sub(r"\\[a-zA-Z]+\*?", " ", text)

    # Remove LaTeX braces.
    text = re.sub(r"[{}]", " ", text)

    # Remove citation patterns such as [12] or [1,2,3].
    text = re.sub(r"\[[0-9,\-\s]+\]", " ", text)

    # Replace URLs with a space.
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Keep letters/numbers and useful punctuation.
    text = re.sub(
        r"[^a-zA-Z0-9\s.,;:!?()\-/+%]",
        " ",
        text,
    )

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ---------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> None:
    required_columns = [
        "paper_id",
        "title",
        "abstract",
        "authors",
        "year",
        "categories",
        "url",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ---------------------------------------------------------------------
# Duplicate removal
# ---------------------------------------------------------------------

def normalize_title_for_duplicate_check(title: str) -> str:
    """
    Normalize titles so trivial formatting differences do not create
    separate papers.
    """
    title = clean_text(title).lower()

    # Remove all non-alphanumeric characters.
    title = re.sub(r"[^a-z0-9]+", "", title)

    return title


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates by:
    1. paper_id
    2. normalized title

    This is intentionally lightweight so the preprocessing step remains
    fast for a large arXiv export.
    """

    df = df.copy()

    before = len(df)

    df["paper_id"] = (
        df["paper_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # First remove exact ID duplicates.
    df = df.drop_duplicates(
        subset=["paper_id"],
        keep="first",
    )

    # Build normalized title.
    df["_normalized_title"] = (
        df["title"]
        .fillna("")
        .astype(str)
        .map(normalize_title_for_duplicate_check)
    )

    # Remove duplicate normalized titles.
    df = df.drop_duplicates(
        subset=["_normalized_title"],
        keep="first",
    )

    df = df.drop(
        columns=["_normalized_title"],
        errors="ignore",
    )

    after = len(df)

    print(
        f"Removed {before - after:,} duplicates. "
        f"Remaining papers: {after:,}"
    )

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    """

    validate_columns(df)

    df = df.copy()

    # Clean fields.
    for column in [
        "title",
        "abstract",
        "authors",
        "categories",
    ]:
        df[column] = (
            df[column]
            .fillna("")
            .map(clean_text)
        )

    # Ensure IDs are strings.
    df["paper_id"] = (
        df["paper_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    # Remove records without a title or abstract.
    df = df[
        df["title"].str.strip().ne("")
    ]

    df = df[
        df["abstract"].str.strip().ne("")
    ]

    # Remove duplicates.
    df = remove_duplicates(df)

    # Combine title and abstract.
    df["combined_text"] = (
        df["title"]
        + " [SEP] "
        + df["abstract"]
    )

    return df.reset_index(drop=True)


def preprocess_file(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
) -> pd.DataFrame:
    """
    Read the unified dataset, preprocess it, and save it.
    """

    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {input_file}\n"
            "Run data_loader.py first."
        )

    print(f"Reading {input_file}...")

    df = pd.read_csv(input_path)

    print(f"Original records: {len(df):,}")

    processed_df = preprocess_dataframe(df)

    processed_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved processed dataset to: {output_file}"
    )

    print(
        f"Final number of papers: "
        f"{len(processed_df):,}"
    )

    return processed_df


# ---------------------------------------------------------------------
# Command-line execution
# ---------------------------------------------------------------------

if __name__ == "__main__":
    try:
        preprocess_file()
    except Exception as exc:
        print(f"ERROR: {exc}")