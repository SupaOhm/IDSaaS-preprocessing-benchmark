from pathlib import Path

import pandas as pd

from src.config import NORMALIZED_LABEL_COLUMN, RAW_DATA_DIR, RAW_LABEL_COLUMN


def normalize_column_name(column: object) -> str:
    """Return a normalized column name."""
    name = str(column).strip().lower()
    name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
    name = name.replace(".", "_").replace("(", "").replace(")", "")

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize DataFrame column names."""
    raw_label_name = normalize_column_name(RAW_LABEL_COLUMN)
    normalized_columns = []

    for column in df.columns:
        normalized = normalize_column_name(column)
        if normalized == raw_label_name:
            normalized = NORMALIZED_LABEL_COLUMN
        normalized_columns.append(normalized)

    df.columns = normalized_columns
    return df


def discover_csv_files(raw_data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Discover CSV files in the raw data directory."""
    return sorted(Path(raw_data_dir).glob("*.csv"))


def load_csv_file(csv_path: Path) -> pd.DataFrame:
    """Load one CSV file with a UTF-8 fallback to latin1."""
    path = Path(csv_path)

    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1")


def load_raw_data() -> pd.DataFrame:
    """Load and concatenate all raw CSV files."""
    csv_files = discover_csv_files(RAW_DATA_DIR)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {RAW_DATA_DIR}")

    frames = []
    for csv_file in csv_files:
        df = load_csv_file(csv_file)
        df = normalize_column_names(df)
        df["source_file"] = csv_file.name
        frames.append(df)

    return pd.concat(frames, ignore_index=True)
