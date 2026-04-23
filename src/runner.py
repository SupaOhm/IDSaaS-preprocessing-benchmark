import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
from pyspark.sql import SparkSession

from src.config import BENIGN_TARGET, INTERMEDIATE_DIR, NORMALIZED_LABEL_COLUMN, RANDOM_SEED
from src.evaluate import calculate_metrics
from src.load_data import load_raw_data
from src.rf_pipeline import run_rf_anomaly_pipeline
from src.spark_pipeline.pipeline import run_preprocessing_pipeline
from src.split import split_train_test
from src.variants import VARIANT_STAGES, get_variant_stages


MODEL_OUTPUT_DIR = INTERMEDIATE_DIR / "models"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run IDSaaS anomaly-branch benchmark.")
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
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[RANDOM_SEED],
        help="One or more random seeds to run.",
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
        SparkSession.builder.appName("IDSaaSAnomalyBranchBenchmark")
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


def save_trained_model(model, variant_name: str, seed: int) -> Path:
    """Save a trained anomaly model."""
    model_path = MODEL_OUTPUT_DIR / f"{variant_name}_seed{seed}.joblib"
    return model.save(model_path)


def print_variant_summary(
    variant_name: str,
    seed: int,
    train_shape_before: tuple[int, int],
    val_shape_before: tuple[int, int],
    test_shape_before: tuple[int, int],
    train_shape_after: tuple[int, int],
    val_shape_after: tuple[int, int],
    test_shape_after: tuple[int, int],
    metrics: dict[str, float | int | None],
    results: dict,
    model_path: Path,
) -> None:
    """Print a comparable experiment summary."""
    print(f"variant: {variant_name}")
    print("model: rf_anomaly")
    print(f"seed: {seed}")
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
    print(f"roc_auc: {metrics['roc_auc']}")
    print(f"pr_auc: {metrics['pr_auc']}")
    print(f"tn: {metrics['tn']}")
    print(f"fp: {metrics['fp']}")
    print(f"fn: {metrics['fn']}")
    print(f"tp: {metrics['tp']}")
    print(f"threshold: {results['threshold']}")
    print(f"derived_threshold: {results['derived_threshold']}")
    print(f"training_time: {results['training_time']}")
    print(f"validation_scoring_time: {results['validation_scoring_time']}")
    print(f"test_scoring_time: {results['test_scoring_time']}")
    print(f"model_info: {results['model_info']}")
    print(f"model_path: {model_path}")
    print()


def run_variant(
    variant_name: str,
    seed: int,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    spark: Optional[SparkSession] = None,
) -> None:
    """Run one preprocessing variant for the anomaly branch."""
    print(f"[runner] starting variant: {variant_name} (seed={seed})")
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
    results = run_rf_anomaly_pipeline(
        X_train_processed,
        X_val_processed,
        X_test_processed,
        X_train_benign,
        X_val_benign,
        random_state=seed,
    )
    print("[runner] RF anomaly pipeline completed")
    print("[runner] saving trained model")
    model_path = save_trained_model(results["model"], variant_name, seed)
    print(f"[runner] model saved to {model_path}")
    metrics = calculate_metrics(y_test_processed, results["y_pred"], scores=results["scores"])

    print("[runner] printing final summary")
    print_variant_summary(
        variant_name,
        seed,
        train_shape_before,
        val_shape_before,
        test_shape_before,
        X_train_processed.shape,
        X_val_processed.shape,
        X_test_processed.shape,
        metrics,
        results,
        model_path,
    )


def main() -> None:
    """Run the selected anomaly-branch variants."""
    args = parse_args()
    variants = requested_variants(args)
    print("[runner] loading raw data")
    df = load_raw_data()
    needs_spark = any(any(get_variant_stages(variant).values()) for variant in variants)
    spark = create_spark_session() if needs_spark else None

    try:
        for seed in args.seeds:
            print(f"[runner] splitting data for seed {seed}")
            splits = split_train_test(df, random_state=seed)
            X_train = splits["X_train"]
            y_train = splits["y_train"]
            X_val = splits["X_val"]
            y_val = splits["y_val"]
            X_test = splits["X_test"]
            y_test = splits["y_test"]

            for variant_name in variants:
                run_variant(
                    variant_name,
                    seed,
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
