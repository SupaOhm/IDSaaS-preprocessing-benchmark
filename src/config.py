from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
INTERMEDIATE_DIR = REPO_ROOT / "outputs" / "intermediate"
LOGS_DIR = REPO_ROOT / "outputs" / "logs"

RANDOM_SEED = 42
TEST_SIZE = 0.2

RAW_LABEL_COLUMN = " Label"
NORMALIZED_LABEL_COLUMN = "label"

BENIGN_LABEL = "BENIGN"
BENIGN_TARGET = 0
ATTACK_TARGET = 1

EXPERIMENT_VARIANTS = (
    "baseline",
    "parse",
    "parse_norm",
    "parse_norm_clean",
)


@dataclass
class RFConfig:
    n_svd_components: int = 64
    n_rff_components: int = 192
    rff_gamma: float = 0.10
    n_rotations: int = 8
    n_estimators: int = 200
    max_depth: int | None = 20
    min_samples_leaf: int = 2
    n_jobs: int = 8
    threshold_quantile: float = 0.97
    calibrated_threshold: float | None = 0.8042
    use_calibrated_threshold: bool = False
    exclude_columns: tuple[str, ...] = (
        "Flow ID",
        "flow_id",
        "Source IP",
        "source_ip",
        "Destination IP",
        "destination_ip",
        "Timestamp",
        "timestamp",
        "SimillarHTTP",
        "simillarhttp",
    )
