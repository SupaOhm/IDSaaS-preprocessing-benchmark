import pandas as pd

from src.config import (
    ATTACK_TARGET,
    BENIGN_LABEL,
    BENIGN_TARGET,
    NORMALIZED_LABEL_COLUMN,
)


SOURCE_FILE_COLUMN = "source_file"
WEEKDAY_ORDER = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")
TRAIN_DAYS = ("Monday", "Tuesday")
VAL_DAYS = ("Wednesday",)
TEST_DAYS = ("Thursday", "Friday")


def build_binary_target(df: pd.DataFrame) -> pd.Series:
    """Convert labels into a binary target series."""
    if NORMALIZED_LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {NORMALIZED_LABEL_COLUMN}")

    labels = df[NORMALIZED_LABEL_COLUMN].astype(str).str.strip()
    return labels.eq(BENIGN_LABEL).map(
        {True: BENIGN_TARGET, False: ATTACK_TARGET}
    )


def group_source_files_by_weekday(df: pd.DataFrame) -> dict[str, list[str]]:
    """Group source files by weekday name detected from filenames."""
    if SOURCE_FILE_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {SOURCE_FILE_COLUMN}")

    weekday_groups = {weekday: [] for weekday in WEEKDAY_ORDER}
    ordered_files = list(dict.fromkeys(df[SOURCE_FILE_COLUMN].astype(str)))

    for source_file in ordered_files:
        matched_weekday = next(
            (weekday for weekday in WEEKDAY_ORDER if weekday.lower() in source_file.lower()),
            None,
        )
        if matched_weekday is not None:
            weekday_groups[matched_weekday].append(source_file)

    missing_days = [weekday for weekday in WEEKDAY_ORDER if not weekday_groups[weekday]]
    if missing_days:
        missing = ", ".join(missing_days)
        raise ValueError(f"Missing required weekday groups in source files: {missing}")

    return weekday_groups


def collect_source_files(
    weekday_groups: dict[str, list[str]],
    weekdays: tuple[str, ...],
) -> list[str]:
    """Collect source files for the requested weekdays."""
    source_files = []
    for weekday in weekdays:
        source_files.extend(weekday_groups[weekday])

    return source_files


def subset_by_source_files(
    X: pd.DataFrame,
    y: pd.Series,
    source_files: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Return rows that belong to the provided source files."""
    mask = X[SOURCE_FILE_COLUMN].astype(str).isin(source_files)
    return X.loc[mask].copy(), y.loc[mask].copy()


def split_train_val_test(
    df: pd.DataFrame,
    **_,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split data into train, validation, test, and benign-only subsets."""
    y = build_binary_target(df)
    X = df.drop(columns=[NORMALIZED_LABEL_COLUMN])

    weekday_groups = group_source_files_by_weekday(X)
    train_files = collect_source_files(weekday_groups, TRAIN_DAYS)
    val_files = collect_source_files(weekday_groups, VAL_DAYS)
    test_files = collect_source_files(weekday_groups, TEST_DAYS)
    X_train, y_train = subset_by_source_files(X, y, train_files)
    X_val, y_val = subset_by_source_files(X, y, val_files)
    X_test, y_test = subset_by_source_files(X, y, test_files)

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_benign": X_train.loc[y_train == BENIGN_TARGET].copy(),
        "X_val_benign": X_val.loc[y_val == BENIGN_TARGET].copy(),
    }


def split_train_test(
    df: pd.DataFrame,
    **_,
) -> dict[str, pd.DataFrame | pd.Series]:
    """Split data using the train, validation, and test split design."""
    return split_train_val_test(df)
