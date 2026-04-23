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
