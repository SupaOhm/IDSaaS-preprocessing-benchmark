from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_pred, scores=None) -> dict[str, float | int | None]:
    """Calculate classification metrics and optional score-based metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "roc_auc": None,
        "pr_auc": None,
    }

    if scores is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError:
            metrics["roc_auc"] = None

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, scores))
        except ValueError:
            metrics["pr_auc"] = None

    return metrics
