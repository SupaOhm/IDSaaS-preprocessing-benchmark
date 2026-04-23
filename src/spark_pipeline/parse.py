from pyspark.sql import DataFrame


def normalize_column_name(column_name: object) -> str:
    """Return a normalized Spark column name."""
    name = str(column_name).strip().lower()
    name = name.replace(" ", "_").replace("/", "_").replace("-", "_")
    name = name.replace(".", "_").replace("(", "").replace(")", "")

    while "__" in name:
        name = name.replace("__", "_")

    return name.strip("_")


def rename_columns(df: DataFrame) -> DataFrame:
    """Rename all DataFrame columns using normalized names."""
    return df.toDF(*(normalize_column_name(column) for column in df.columns))


def parse_dataframe(df: DataFrame) -> DataFrame:
    """Apply parse-stage structural preparation."""
    return rename_columns(df)
