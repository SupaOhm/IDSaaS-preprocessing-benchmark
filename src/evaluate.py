from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_metrics(y_true, y_pred) -> dict[str, float]:
    """Calculate binary classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
