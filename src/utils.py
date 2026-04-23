import random

import numpy as np
import pandas as pd


def set_random_seed(seed: int) -> None:
    """Set Python and NumPy random seeds."""
    random.seed(seed)
    np.random.seed(seed)


def canonicalize_column_list(columns) -> list[str]:
    """Return column names in a normalized canonical form."""
    canonical_columns = []

    for column in columns:
        name = str(column).strip().lower()
        name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
        name = name.replace(".", "_").replace("(", "").replace(")", "")

        while "__" in name:
            name = name.replace("__", "_")

        canonical_columns.append(name.strip("_"))

    return canonical_columns


def infer_numeric_and_categorical(
    df: pd.DataFrame,
    feature_columns,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature columns from pandas dtypes."""
    numeric_columns = []
    categorical_columns = []

    for column in feature_columns:
        if pd.api.types.is_bool_dtype(df[column]):
            categorical_columns.append(column)
        elif pd.api.types.is_numeric_dtype(df[column]):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)

    return numeric_columns, categorical_columns
