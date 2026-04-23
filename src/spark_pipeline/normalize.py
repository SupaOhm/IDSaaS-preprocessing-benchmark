from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def normalize_string_column(column: Column, uppercase: bool = False) -> Column:
    """Trim a string column and optionally uppercase it."""
    normalized = F.trim(column)
    if uppercase:
        normalized = F.upper(normalized)

    return normalized


def normalize_dataframe(df: DataFrame) -> DataFrame:
    """Apply normalize-stage transformations."""
    if "label" not in df.columns:
        return df

    label_field = next(field for field in df.schema.fields if field.name == "label")
    if not isinstance(label_field.dataType, StringType):
        return df

    return df.withColumn("label", normalize_string_column(F.col("label"), uppercase=True))
