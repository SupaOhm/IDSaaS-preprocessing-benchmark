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


def build_binary_target(df: pd.DataFrame) -> pd.Series:
    """Convert labels into a binary target series."""
    if NORMALIZED_LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column: {NORMALIZED_LABEL_COLUMN}")

    return df[NORMALIZED_LABEL_COLUMN].eq(BENIGN_LABEL).map(
        {True: BENIGN_TARGET, False: ATTACK_TARGET}
    )


def split_train_test(df: pd.DataFrame):
    """Split features and binary target into train and test sets."""
    y = build_binary_target(df)
    X = df.drop(columns=[NORMALIZED_LABEL_COLUMN])

    stratify = y if y.nunique() > 1 and y.value_counts().min() >= 2 else None

    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=stratify,
    )
