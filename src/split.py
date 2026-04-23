import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ATTACK_TARGET,
    BENIGN_LABEL,
    BENIGN_TARGET,
    NORMALIZED_LABEL_COLUMN,
    RANDOM_SEED,
    TEST_SIZE,
)


VAL_SIZE = 0.2


def build_binary_target(df: pd.DataFrame) -> pd.Series:
    """Convert labels into a binary target series."""
    if NORMALIZED_LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {NORMALIZED_LABEL_COLUMN}")

    labels = df[NORMALIZED_LABEL_COLUMN].astype(str).str.strip()
    return labels.eq(BENIGN_LABEL).map(
        {True: BENIGN_TARGET, False: ATTACK_TARGET}
    )


def stratify_when_valid(y: pd.Series) -> pd.Series | None:
    """Return a stratify target when class counts allow it."""
    return y if y.nunique() > 1 and y.value_counts().min() >= 2 else None


def split_train_val_test(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Split data into train, validation, test, and benign-only subsets."""
    y = build_binary_target(df)
    X = df.drop(columns=[NORMALIZED_LABEL_COLUMN])

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify_when_valid(y),
    )

    relative_val_size = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=RANDOM_SEED,
        stratify=stratify_when_valid(y_train_val),
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "X_train_benign": X_train.loc[y_train == BENIGN_TARGET],
        "X_val_benign": X_val.loc[y_val == BENIGN_TARGET],
    }


def split_train_test(df: pd.DataFrame) -> dict[str, pd.DataFrame | pd.Series]:
    """Split data using the train, validation, and test split design."""
    return split_train_val_test(df)
