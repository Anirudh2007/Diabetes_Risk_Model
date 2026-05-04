"""
Evaluation module: metrics, threshold optimization, and statistical tests.

Provides:
    - Youden's J statistic threshold optimization
    - 7 performance metrics (accuracy, sensitivity, specificity, precision,
      F1, AUC-ROC, confusion matrix)
    - Welch's t-tests on all 8 features
    - Feature–outcome correlation analysis
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURE_NAMES,
    SIGNIFICANCE_ALPHA,
    THRESHOLD_SEARCH_POINTS,
    OUTPUT_DIR,
)


def find_optimal_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_points: int = THRESHOLD_SEARCH_POINTS,
) -> dict:
    """
    Find the optimal decision threshold using Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1

    The threshold that maximises J gives the best trade-off between
    sensitivity (catching diabetics) and specificity (avoiding false alarms).

    Args:
        y_true: True binary labels, shape (n_samples,).
        probabilities: Predicted P(diabetic), shape (n_samples,).
        n_points: Number of candidate thresholds to evaluate.

    Returns:
        Dictionary with:
            optimal_threshold: Best threshold value.
            optimal_sensitivity: Sensitivity at optimal threshold.
            optimal_specificity: Specificity at optimal threshold.
            optimal_j_statistic: Maximum J value achieved.
            thresholds: Array of all candidate thresholds.
            sensitivities: Sensitivity at each threshold.
            specificities: Specificity at each threshold.
            f1_scores: F1 score at each threshold.
    """
    thresholds = np.linspace(0.01, 0.99, n_points)
    sensitivities = np.zeros(n_points)
    specificities = np.zeros(n_points)
    f1_scores = np.zeros(n_points)

    for index, threshold in enumerate(thresholds):
        predictions = (probabilities >= threshold).astype(int)

        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        # Sensitivity = TP / (TP + FN)  — true positive rate
        sensitivities[index] = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Specificity = TN / (TN + FP)  — true negative rate
        specificities[index] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # F1 Score = 2·(Precision·Recall) / (Precision + Recall)
        f1_scores[index] = f1_score(y_true, predictions, zero_division=0)

    # Youden's J = Sensitivity + Specificity − 1
    j_statistics = sensitivities + specificities - 1
    optimal_index = np.argmax(j_statistics)

    return {
        "optimal_threshold": thresholds[optimal_index],
        "optimal_sensitivity": sensitivities[optimal_index],
        "optimal_specificity": specificities[optimal_index],
        "optimal_j_statistic": j_statistics[optimal_index],
        "thresholds": thresholds,
        "sensitivities": sensitivities,
        "specificities": specificities,
        "f1_scores": f1_scores,
    }


def compute_performance_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict:
    """
    Compute the full suite of 7 performance metrics on a test set.

    Args:
        y_true: True binary labels.
        probabilities: Predicted P(diabetic).
        threshold: Decision threshold (from Youden's J optimization).

    Returns:
        Dictionary with accuracy, sensitivity, specificity, precision,
        f1_score, auc_roc, and confusion_matrix components.
    """
    predictions = (probabilities >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    accuracy = accuracy_score(y_true, predictions)
    sensitivity = recall_score(y_true, predictions, zero_division=0)  # = Recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = precision_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)
    auc_roc = roc_auc_score(y_true, probabilities)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1,
        "auc_roc": auc_roc,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def perform_welch_t_tests(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list = FEATURE_NAMES,
    alpha: float = SIGNIFICANCE_ALPHA,
) -> pd.DataFrame:
    """
    Perform Welch's t-tests comparing diabetic vs non-diabetic groups.

    Welch's t-test does not assume equal variances and is more robust
    than Student's t-test for clinical datasets.

    Args:
        X_train: Training feature matrix, shape (n, 8).
        y_train: Training labels, shape (n,).
        feature_names: Names of the 8 features.
        alpha: Significance level (default 0.05).

    Returns:
        DataFrame with columns: Feature, t_statistic, p_value,
        Significant, Stars.
    """
    results = []

    for feature_index, feature_name in enumerate(feature_names):
        # Split feature values by class
        values_class0 = X_train[y_train == 0, feature_index]
        values_class1 = X_train[y_train == 1, feature_index]

        # Welch's t-test (equal_var=False)
        t_statistic, p_value = ttest_ind(
            values_class0, values_class1, equal_var=False
        )

        # Significance level and stars
        if p_value < 0.001:
            stars = "***"
        elif p_value < 0.01:
            stars = "**"
        elif p_value < alpha:
            stars = "*"
        else:
            stars = ""

        significant = "YES" if p_value < alpha else "NO"

        results.append({
            "Feature": feature_name,
            "t_statistic": round(t_statistic, 4),
            "p_value": p_value,
            "Significant": significant,
            "Stars": stars,
        })

    results_df = pd.DataFrame(results)

    # Save to CSV
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "t_test_results.csv")
    results_df.to_csv(csv_path, index=False)

    return results_df


def compute_feature_correlations(dataframe: pd.DataFrame) -> pd.Series:
    """
    Calculate Pearson correlation of each feature with the Outcome variable.

    Args:
        dataframe: Cleaned DataFrame with features and Outcome.

    Returns:
        Series of correlation values sorted by absolute magnitude (descending).
    """
    correlations = dataframe[FEATURE_NAMES + ["Outcome"]].corr()["Outcome"].drop("Outcome")
    return correlations.reindex(correlations.abs().sort_values(ascending=False).index)


def get_roc_curve_data(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> tuple:
    """
    Compute ROC curve coordinates for plotting.

    Args:
        y_true: True binary labels.
        probabilities: Predicted P(diabetic).

    Returns:
        Tuple of (fpr_array, tpr_array, thresholds_array).
    """
    fpr, tpr, thresholds = roc_curve(y_true, probabilities)
    return fpr, tpr, thresholds
