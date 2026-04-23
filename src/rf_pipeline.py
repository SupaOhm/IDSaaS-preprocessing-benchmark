import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_SEED, RFConfig
from src.utils import (
    canonicalize_column_list,
    infer_numeric_and_categorical,
    set_random_seed,
)


DEFAULT_NON_FEATURE_COLUMNS = {
    "binary_label",
    "multiclass_label",
    "source_file",
    "row_id",
}


def select_feature_columns(df, exclude_columns=()) -> list[str]:
    """Select feature columns after applying canonical exclusions."""
    excluded = set(canonicalize_column_list(DEFAULT_NON_FEATURE_COLUMNS))
    excluded.update(canonicalize_column_list(exclude_columns))

    feature_columns = [
        column
        for column, canonical_column in zip(df.columns, canonicalize_column_list(df.columns))
        if canonical_column not in excluded
    ]

    if not feature_columns:
        raise ValueError("No feature columns remain after exclusions.")

    return feature_columns


def _make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_tabular_preprocessor(
    df,
    feature_columns,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    """Build a tabular preprocessor for numeric and categorical features."""
    numeric_columns, categorical_columns = infer_numeric_and_categorical(
        df,
        feature_columns,
    )

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", _make_one_hot_encoder()),
                    ]
                ),
                categorical_columns,
            )
        )

    return ColumnTransformer(transformers=transformers)


def sanitize_numeric_features(df, feature_columns):
    """Replace infinite numeric values with NaN in selected feature columns."""
    sanitized = df.copy()
    numeric_columns, _ = infer_numeric_and_categorical(sanitized, feature_columns)

    if numeric_columns:
        sanitized.loc[:, numeric_columns] = sanitized.loc[:, numeric_columns].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return sanitized


class SelfSupervisedRFAnomaly:
    """Self-supervised benign-only RF anomaly detector."""

    def __init__(
        self,
        config: RFConfig | None = None,
        random_state: int = RANDOM_SEED,
        scale_numeric: bool = True,
        verbose: bool = True,
    ):
        self.config = config or RFConfig()
        self.random_state = random_state
        self.scale_numeric = scale_numeric
        self.verbose = verbose
        self.feature_columns_ = None
        self.preprocessor_ = None
        self.svd_ = None
        self.rff_ = None
        self.rotations_ = None
        self.rf_ = None
        self.threshold_ = None
        self.derived_threshold_ = None

    def _log(self, message: str) -> None:
        """Print a short progress message when verbose logging is enabled."""
        if self.verbose:
            print(message)

    def _to_dense(self, X) -> np.ndarray:
        """Convert sparse or array-like data to a dense NumPy array."""
        if hasattr(X, "toarray"):
            return X.toarray()

        return np.asarray(X)

    def _make_rotations(self, n_features: int) -> list[np.ndarray]:
        """Create random orthonormal rotation matrices."""
        rng = np.random.default_rng(self.random_state)
        rotations = [np.eye(n_features, dtype=np.float32)]

        for _ in range(max(0, self.config.n_rotations - 1)):
            matrix = rng.normal(size=(n_features, n_features))
            rotation, _ = np.linalg.qr(matrix)
            if np.linalg.det(rotation) < 0:
                rotation[:, 0] *= -1.0
            rotations.append(rotation.astype(np.float32))

        return rotations

    def _apply_transform_chain(self, X) -> np.ndarray:
        """Apply preprocessor, SVD, and RFF transforms."""
        sanitized = sanitize_numeric_features(X, self.feature_columns_)
        transformed = self.preprocessor_.transform(sanitized[self.feature_columns_])
        transformed = self.svd_.transform(transformed)
        transformed = self.rff_.transform(transformed)
        return self._to_dense(transformed).astype(np.float32)

    def _build_self_supervised_dataset(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build rotated self-supervised samples and rotation labels."""
        rotated_features = []
        rotation_labels = []

        for label, rotation in enumerate(self.rotations_):
            rotated_features.append(X @ rotation)
            rotation_labels.append(np.full(X.shape[0], label, dtype=np.int32))

        return np.vstack(rotated_features), np.concatenate(rotation_labels)

    def fit(self, X_train_benign, X_val_benign):
        """Fit the benign-only self-supervised anomaly model."""
        set_random_seed(self.random_state)
        train_benign = X_train_benign
        val_benign = X_val_benign

        if len(train_benign) > 450000:
            train_benign = train_benign.sample(
                n=450000,
                random_state=self.random_state,
            )
        if len(val_benign) > 120000:
            val_benign = val_benign.sample(
                n=120000,
                random_state=self.random_state,
            )

        self.feature_columns_ = select_feature_columns(
            train_benign,
            self.config.exclude_columns,
        )
        self.preprocessor_ = build_tabular_preprocessor(
            train_benign,
            self.feature_columns_,
            scale_numeric=self.scale_numeric,
        )

        self._log("Fitting RF anomaly preprocessor")
        train_benign = sanitize_numeric_features(train_benign, self.feature_columns_)
        train_prepared = self.preprocessor_.fit_transform(train_benign[self.feature_columns_])
        svd_components = min(
            self.config.n_svd_components,
            max(2, train_prepared.shape[1] - 1),
        )

        self.svd_ = TruncatedSVD(
            n_components=svd_components,
            random_state=self.random_state,
        )
        train_svd = self.svd_.fit_transform(train_prepared)

        self.rff_ = RBFSampler(
            gamma=self.config.rff_gamma,
            n_components=self.config.n_rff_components,
            random_state=self.random_state,
        )
        train_rff = self._to_dense(self.rff_.fit_transform(train_svd)).astype(np.float32)

        self.rotations_ = self._make_rotations(train_rff.shape[1])
        X_self_supervised, y_self_supervised = self._build_self_supervised_dataset(train_rff)

        self.rf_ = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            n_jobs=self.config.n_jobs,
            random_state=self.random_state,
        )

        self._log("Fitting self-supervised RF anomaly model")
        self.rf_.fit(X_self_supervised, y_self_supervised)

        val_scores = self.score_samples(val_benign)
        self.derived_threshold_ = float(
            np.quantile(val_scores, self.config.threshold_quantile)
        )
        if self.config.use_calibrated_threshold:
            self.threshold_ = float(self.config.calibrated_threshold)
        else:
            self.threshold_ = self.derived_threshold_

        return self

    def score_samples(self, X) -> np.ndarray:
        """Return anomaly scores where larger values are more anomalous."""
        transformed = self._apply_transform_chain(X)
        expected_probabilities = []

        for label, rotation in enumerate(self.rotations_):
            rotated = transformed @ rotation
            probabilities = self.rf_.predict_proba(rotated)
            class_index = np.where(self.rf_.classes_ == label)[0]

            if len(class_index) == 0:
                expected_probabilities.append(np.zeros(rotated.shape[0]))
            else:
                expected_probabilities.append(probabilities[:, class_index[0]])

        mean_expected_probability = np.mean(expected_probabilities, axis=0)
        return 1.0 - mean_expected_probability

    def predict(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict binary anomaly labels using the fitted threshold."""
        if self.threshold_ is None:
            raise ValueError("Model must be fitted before prediction.")

        scores = self.score_samples(X)
        preds = (scores > self.threshold_).astype(int)
        return preds, scores


def run_rf_anomaly_pipeline(
    X_train,
    X_val,
    X_test,
    X_train_benign,
    X_val_benign,
    config: RFConfig | None = None,
    random_state: int = RANDOM_SEED,
) -> dict:
    """Fit the RF anomaly detector and predict on the test set."""
    model = SelfSupervisedRFAnomaly(
        config=config,
        random_state=random_state,
    )
    model.fit(X_train_benign, X_val_benign)

    y_pred, scores = model.predict(X_test)

    return {
        "model": model,
        "y_pred": y_pred,
        "scores": scores,
        "threshold": model.threshold_,
        "derived_threshold": model.derived_threshold_,
    }
