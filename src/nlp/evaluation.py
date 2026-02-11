from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute standard classification metrics for binary classification.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> np.ndarray:
    """
    Compute confusion matrix.
    """
    return confusion_matrix(y_true, y_pred)


def evaluate_model(
    model,
    X,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Full evaluation pipeline:
    - predict
    - compute metrics
    - return results
    """
    y_pred = model.predict(X)

    metrics = compute_classification_metrics(y, y_pred)
    return metrics


def get_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """
    Return detailed sklearn classification report.
    """
    return classification_report(y_true, y_pred)
