from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ByteType,
    DecimalType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    ShortType,
)


NUMERIC_TYPES = (
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    DecimalType,
)
FLOAT_TYPES = (FloatType, DoubleType)


def get_numeric_columns(df: DataFrame) -> list[str]:
    """Return numeric column names from the DataFrame schema."""
    return [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NUMERIC_TYPES)
    ]


def clean_dataframe(df: DataFrame) -> DataFrame:
    """Apply conservative clean-stage transformations."""
    cleaned = df

    if "label" in cleaned.columns:
        cleaned = cleaned.filter(F.col("label").isNotNull())

    numeric_columns = get_numeric_columns(cleaned)

    for field in cleaned.schema.fields:
        if field.name in numeric_columns and isinstance(field.dataType, FLOAT_TYPES):
            column = F.col(field.name)
            cleaned = cleaned.withColumn(
                field.name,
                F.when(
                    F.isnan(column)
                    | (column == float("inf"))
                    | (column == float("-inf")),
                    F.lit(None),
                ).otherwise(column),
            )

    if numeric_columns:
        cleaned = cleaned.dropna(subset=numeric_columns)

    return cleaned
