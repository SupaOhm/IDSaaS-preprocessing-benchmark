import argparse
from typing import Optional

import pandas as pd
from pyspark.sql import SparkSession

from src.config import BENIGN_TARGET, NORMALIZED_LABEL_COLUMN
from src.evaluate import calculate_metrics
from src.load_data import load_raw_data
from src.rf_pipeline import run_rf_anomaly_pipeline
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
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    stage_config: dict[str, bool],
):
    """Run Spark preprocessing while keeping labels aligned."""
    print("[runner] starting Spark preprocessing for train")
    train_spark = spark.createDataFrame(attach_labels(X_train, y_train))
    print("[runner] starting Spark preprocessing for val")
    val_spark = spark.createDataFrame(attach_labels(X_val, y_val))
    print("[runner] starting Spark preprocessing for test")
    test_spark = spark.createDataFrame(attach_labels(X_test, y_test))

    train_processed = run_preprocessing_pipeline(
        train_spark,
        stage_config,
    ).toPandas()
    print("[runner] finished Spark preprocessing for train")
    val_processed = run_preprocessing_pipeline(
        val_spark,
        stage_config,
    ).toPandas()
    print("[runner] finished Spark preprocessing for val")
    test_processed = run_preprocessing_pipeline(
        test_spark,
        stage_config,
    ).toPandas()
    print("[runner] finished Spark preprocessing for test")

    return (
        split_features_and_target(train_processed),
        split_features_and_target(val_processed),
        split_features_and_target(test_processed),
    )


def attach_labels(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Attach the target series as the normalized label column."""
    df = X.copy()
    df[NORMALIZED_LABEL_COLUMN] = y.values
    return df


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a processed DataFrame back into features and numeric target."""
    if NORMALIZED_LABEL_COLUMN not in df.columns:
        raise ValueError(f"Missing required column after preprocessing: {NORMALIZED_LABEL_COLUMN}")

    y = pd.to_numeric(df[NORMALIZED_LABEL_COLUMN], errors="raise")
    X = df.drop(columns=[NORMALIZED_LABEL_COLUMN])
    return X, y


def print_variant_summary(
    variant_name: str,
    train_shape_before: tuple[int, int],
    val_shape_before: tuple[int, int],
    test_shape_before: tuple[int, int],
    train_shape_after: tuple[int, int],
    val_shape_after: tuple[int, int],
    test_shape_after: tuple[int, int],
    metrics: dict[str, float],
    threshold: float,
    derived_threshold: float,
) -> None:
    """Print a simple variant summary."""
    print(f"variant: {variant_name}")
    print(f"X_train before preprocessing: {train_shape_before}")
    print(f"X_val before preprocessing: {val_shape_before}")
    print(f"X_test before preprocessing: {test_shape_before}")
    print(f"X_train after preprocessing: {train_shape_after}")
    print(f"X_val after preprocessing: {val_shape_after}")
    print(f"X_test after preprocessing: {test_shape_after}")
    print(f"accuracy: {metrics['accuracy']}")
    print(f"precision: {metrics['precision']}")
    print(f"recall: {metrics['recall']}")
    print(f"f1: {metrics['f1']}")
    print(f"threshold: {threshold}")
    print(f"derived_threshold: {derived_threshold}")
    print()


def run_variant(
    variant_name: str,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    spark: Optional[SparkSession] = None,
) -> None:
    """Run one preprocessing variant and print its summary."""
    print(f"[runner] starting variant: {variant_name}")
    train_shape_before = X_train.shape
    val_shape_before = X_val.shape
    test_shape_before = X_test.shape
    stage_config = get_variant_stages(variant_name)

    if any(stage_config.values()):
        print("[runner] variant uses Spark preprocessing")
        (
            (X_train_processed, y_train_processed),
            (X_val_processed, y_val_processed),
            (X_test_processed, y_test_processed),
        ) = preprocess_with_spark(
            spark,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            stage_config,
        )
    else:
        print("[runner] variant uses baseline preprocessing")
        X_train_processed = X_train
        y_train_processed = y_train
        X_val_processed = X_val
        y_val_processed = y_val
        X_test_processed = X_test
        y_test_processed = y_test

    print("[runner] building benign subsets")
    X_train_benign = X_train_processed.loc[y_train_processed == BENIGN_TARGET]
    X_val_benign = X_val_processed.loc[y_val_processed == BENIGN_TARGET]

    print("[runner] starting RF anomaly pipeline")
    rf_results = run_rf_anomaly_pipeline(
        X_train_processed,
        X_val_processed,
        X_test_processed,
        X_train_benign,
        X_val_benign,
    )
    print("[runner] RF anomaly pipeline completed")
    metrics = calculate_metrics(y_test_processed, rf_results["y_pred"])

    print("[runner] printing final summary")
    print_variant_summary(
        variant_name,
        train_shape_before,
        val_shape_before,
        test_shape_before,
        X_train_processed.shape,
        X_val_processed.shape,
        X_test_processed.shape,
        metrics,
        rf_results["threshold"],
        rf_results["derived_threshold"],
    )


def main() -> None:
    """Run the selected preprocessing variant summaries."""
    args = parse_args()
    variants = requested_variants(args)
    print("[runner] loading raw data")
    df = load_raw_data()
    print("[runner] splitting data")
    splits = split_train_test(df)
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]
    needs_spark = any(any(get_variant_stages(variant).values()) for variant in variants)
    spark = create_spark_session() if needs_spark else None

    try:
        for variant_name in variants:
            run_variant(
                variant_name,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                spark,
            )
    finally:
        if spark is not None:
            spark.stop()


if __name__ == "__main__":
    main()
