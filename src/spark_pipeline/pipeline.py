from pyspark.sql import DataFrame

from src.spark_pipeline.clean import clean_dataframe
from src.spark_pipeline.normalize import normalize_dataframe
from src.spark_pipeline.parse import parse_dataframe


def run_preprocessing_pipeline(df: DataFrame, stage_config: dict[str, bool]) -> DataFrame:
    """Apply enabled Spark preprocessing stages in order."""
    transformed = df

    if stage_config.get("parse", False):
        transformed = parse_dataframe(transformed)

    if stage_config.get("normalize", False):
        transformed = normalize_dataframe(transformed)

    if stage_config.get("clean", False):
        transformed = clean_dataframe(transformed)

    return transformed
