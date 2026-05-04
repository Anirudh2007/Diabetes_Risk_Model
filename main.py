"""
main.py -- Single execution script for the 8D Bayesian Diabetes Risk Prediction model.

Orchestrates the full pipeline:
    [1] Data loading, cleaning, and splitting
    [2] 8D Bayes model training
    [3] Performance evaluation on the test set at threshold=0.5

Usage:
    python main.py
"""

import sys
import os
import warnings
import numpy as np

# Suppress non-critical warnings for cleaner console output
warnings.filterwarnings("ignore")

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEATURE_NAMES, DECISION_THRESHOLD
from src.data_cleaning import run_cleaning_pipeline
from src.model import BayesDiabetesClassifier
from src.evaluation import compute_performance_metrics


def print_separator(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n[{title}]")


def main() -> None:
    """Execute the complete 8D Bayesian diabetes prediction pipeline."""

    print("=" * 52)
    print("  DIABETES RISK PREDICTION: 8D BAYES MODEL")
    print("=" * 52)

    # ------------------------------------------------
    # [1] DATA PROCESSING
    # ------------------------------------------------
    print_separator("1] DATA PROCESSING")
    dataframe, X_train, X_test, y_train, y_test = run_cleaning_pipeline()

    # ------------------------------------------------
    # [2] TRAIN-TEST SPLIT (reported by data_cleaning)
    # ------------------------------------------------
    print_separator("2] TRAIN-TEST SPLIT")
    print(f"    [OK] Training: {len(y_train)} samples ({y_train.mean():.1%} diabetic)")
    print(f"    [OK] Testing:  {len(y_test)} samples ({y_test.mean():.1%} diabetic)")

    # ------------------------------------------------
    # [3] MODEL TRAINING
    # ------------------------------------------------
    print_separator("3] MODEL TRAINING")

    model = BayesDiabetesClassifier()
    model.fit(X_train, y_train)

    summary = model.get_model_summary()

    print("    [OK] 8D Bayes model fitted on training data")
    print(f"    [OK] Prior P(Diabetic)     = {summary['prior_diabetic']:.4f}")
    print(f"    [OK] Prior P(Non-Diabetic) = {summary['prior_non_diabetic']:.4f}")
    print("    [OK] Covariance matrices positive definite")
    print(f"      Min eigenvalue (class 0): {summary['min_eigenvalue_class0']:.6e}")
    print(f"      Min eigenvalue (class 1): {summary['min_eigenvalue_class1']:.6e}")

    # Print mean vectors
    print("\n    Mean vector (Non-Diabetic):")
    for name, value in zip(FEATURE_NAMES, summary["mean_non_diabetic"]):
        print(f"      {name:30s} = {value:.4f}")
    print("\n    Mean vector (Diabetic):")
    for name, value in zip(FEATURE_NAMES, summary["mean_diabetic"]):
        print(f"      {name:30s} = {value:.4f}")

    # ------------------------------------------------
    # [4] PERFORMANCE METRICS (Test Set, threshold=0.5)
    # ------------------------------------------------
    print_separator("4] PERFORMANCE METRICS (Test Set, threshold=0.5)")

    test_probabilities = model.predict_probabilities(X_test)
    metrics = compute_performance_metrics(y_test, test_probabilities, DECISION_THRESHOLD)

    print(f"    Decision threshold: {DECISION_THRESHOLD}")
    print(f"    Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
    print(f"    Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']:.1%})")
    print(f"    Specificity: {metrics['specificity']:.4f} ({metrics['specificity']:.1%})")
    print(f"    Precision:   {metrics['precision']:.4f} ({metrics['precision']:.1%})")

    print("\n    Confusion Matrix:")
    print("                        Predicted 0   Predicted 1")
    print(f"      Actual 0 (Healthy)   TN={metrics['true_negatives']:4d}       FP={metrics['false_positives']:4d}")
    print(f"      Actual 1 (Diabetic)  FN={metrics['false_negatives']:4d}       TP={metrics['true_positives']:4d}")

    # ------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------
    print(f"\n{'=' * 52}")
    print("  [OK] Analysis complete.")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    main()
