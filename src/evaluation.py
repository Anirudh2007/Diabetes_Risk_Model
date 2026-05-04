"""
Evaluation module: performance metrics at a fixed decision threshold.

Provides:
    - 4 performance metrics (accuracy, sensitivity, specificity, precision)
      and confusion matrix components evaluated at a fixed threshold (0.5).
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_performance_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute performance metrics on a test set at a fixed decision threshold.

    Args:
        y_true: True binary labels.
        probabilities: Predicted P(diabetic).
        threshold: Decision threshold (default 0.5).

    Returns:
        Dictionary with accuracy, sensitivity, specificity, precision,
        and confusion matrix components.
    """
    predictions = (probabilities >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, predictions)
    sensitivity = recall_score(y_true, predictions, zero_division=0)  # = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, predictions, zero_division=0)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }
