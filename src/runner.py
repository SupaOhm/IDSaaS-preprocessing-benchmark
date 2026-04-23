import argparse
from typing import Optional

from pyspark.sql import SparkSession

from src.load_data import load_raw_data
from src.spark_pipeline.pipeline import run_preprocessing_pipeline
from src.split import split_train_test
from src.variants import VARIANT_STAGES, get_variant_stages


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run IDSaaS preprocessing benchmark.")
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_STAGES),
        default="baseline",
        help="Experiment variant to run.",
    )
    parser.add_argument(
        "--run-all",
        action="store_true",
        help="Run all supported variants.",
    )
    return parser.parse_args()


def requested_variants(args: argparse.Namespace) -> list[str]:
    """Return the variant names requested by the CLI."""
    if args.run_all:
        return list(VARIANT_STAGES)

    return [args.variant]


def create_spark_session() -> SparkSession:
    """Create a local Spark session for preprocessing."""
    return (
        SparkSession.builder.appName("IDSaaSPreprocessingBenchmark")
        .master("local[*]")
        .getOrCreate()
    )


def preprocess_with_spark(
    spark: SparkSession,
    X_train,
    X_test,
    stage_config: dict[str, bool],
):
    """Run Spark preprocessing for train and test feature frames."""
    train_spark = spark.createDataFrame(X_train)
    test_spark = spark.createDataFrame(X_test)

    X_train_processed = run_preprocessing_pipeline(
        train_spark,
        stage_config,
    ).toPandas()
    X_test_processed = run_preprocessing_pipeline(
        test_spark,
        stage_config,
    ).toPandas()

    return X_train_processed, X_test_processed


def print_variant_summary(
    variant_name: str,
    train_shape_before: tuple[int, int],
    test_shape_before: tuple[int, int],
    train_shape_after: tuple[int, int],
    test_shape_after: tuple[int, int],
) -> None:
    """Print a simple variant summary."""
    print(f"variant: {variant_name}")
    print(f"X_train before preprocessing: {train_shape_before}")
    print(f"X_test before preprocessing: {test_shape_before}")
    print(f"X_train after preprocessing: {train_shape_after}")
    print(f"X_test after preprocessing: {test_shape_after}")
    print()


def run_variant(
    variant_name: str,
    X_train,
    X_test,
    spark: Optional[SparkSession] = None,
) -> None:
    """Run one preprocessing variant and print its summary."""
    train_shape_before = X_train.shape
    test_shape_before = X_test.shape
    stage_config = get_variant_stages(variant_name)

    if any(stage_config.values()):
        X_train_processed, X_test_processed = preprocess_with_spark(
            spark,
            X_train,
            X_test,
            stage_config,
        )
    else:
        X_train_processed = X_train
        X_test_processed = X_test

    print_variant_summary(
        variant_name,
        train_shape_before,
        test_shape_before,
        X_train_processed.shape,
        X_test_processed.shape,
    )


def main() -> None:
    """Run the selected preprocessing variant summaries."""
    args = parse_args()
    variants = requested_variants(args)
    df = load_raw_data()
    X_train, X_test, _y_train, _y_test = split_train_test(df)
    needs_spark = any(any(get_variant_stages(variant).values()) for variant in variants)
    spark = create_spark_session() if needs_spark else None

    try:
        for variant_name in variants:
            run_variant(variant_name, X_train, X_test, spark)
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
