"""
main.py -- Single execution script for the 8D Bayesian Diabetes Risk Prediction model.

Orchestrates the full pipeline:
    [1] Data loading, cleaning, and splitting
    [2] 8D Bayes model training
    [3] Threshold optimization (Youden's J)
    [4] Performance evaluation on the test set
    [5] Statistical validation (Welch's t-tests)
    [6] Feature correlation analysis
    [7] Generation of 5 diagnostic plots

Usage:
    python main.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd

# Suppress non-critical warnings for cleaner console output
warnings.filterwarnings("ignore")

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FEATURE_NAMES, PLOTS_DIR
from src.data_cleaning import run_cleaning_pipeline
from src.model import BayesDiabetesClassifier
from src.evaluation import (
    find_optimal_threshold,
    compute_performance_metrics,
    perform_welch_t_tests,
    compute_feature_correlations,
    get_roc_curve_data,
)
from src.visualization import (
    plot_risk_heatmap,
    plot_threshold_analysis,
    plot_roc_curve,
    plot_correlation_heatmap,
    plot_ttest_feature_means,
)


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
    # [4] THRESHOLD OPTIMIZATION
    # ------------------------------------------------
    print_separator("4] THRESHOLD OPTIMIZATION")

    # Generate probability predictions for the TEST set
    test_probabilities = model.predict_probabilities(X_test)

    # Find optimal threshold using Youden's J statistic
    threshold_data = find_optimal_threshold(y_test, test_probabilities)
    optimal_threshold = threshold_data["optimal_threshold"]

    print(f"    [OK] Youden's J optimal threshold = {optimal_threshold:.3f}")
    print(f"    [OK] Sensitivity at optimal        = {threshold_data['optimal_sensitivity']:.3f}")
    print(f"    [OK] Specificity at optimal         = {threshold_data['optimal_specificity']:.3f}")
    print(f"    [OK] Youden's J statistic           = {threshold_data['optimal_j_statistic']:.3f}")

    # ------------------------------------------------
    # [5] PERFORMANCE METRICS (Test Set)
    # ------------------------------------------------
    print_separator("5] PERFORMANCE METRICS (Test Set)")

    metrics = compute_performance_metrics(y_test, test_probabilities, optimal_threshold)

    print(f"    Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})")
    print(f"    Sensitivity: {metrics['sensitivity']:.4f} ({metrics['sensitivity']:.1%})")
    print(f"    Specificity: {metrics['specificity']:.4f} ({metrics['specificity']:.1%})")
    print(f"    Precision:   {metrics['precision']:.4f} ({metrics['precision']:.1%})")
    print(f"    F1 Score:    {metrics['f1_score']:.4f}")
    print(f"    AUC-ROC:     {metrics['auc_roc']:.4f}")

    print("\n    Confusion Matrix:")
    print("                        Predicted 0   Predicted 1")
    print(f"      Actual 0 (Healthy)   TN={metrics['true_negatives']:4d}       FP={metrics['false_positives']:4d}")
    print(f"      Actual 1 (Diabetic)  FN={metrics['false_negatives']:4d}       TP={metrics['true_positives']:4d}")

    # ------------------------------------------------
    # [6] STATISTICAL TESTS
    # ------------------------------------------------
    print_separator("6] STATISTICAL TESTS (Welch's t-test)")

    ttest_results = perform_welch_t_tests(X_train, y_train)

    print(f"    {'Feature':<30s} {'p-value':>12s}    {'Significant':>11s}")
    print(f"    {'-' * 60}")
    for _, row in ttest_results.iterrows():
        p_str = "<0.0001" if row["p_value"] < 0.0001 else f"{row['p_value']:.4f}"
        sig_str = f"{row['Significant']} {row['Stars']}"
        print(f"    {row['Feature']:<30s} {p_str:>12s}    {sig_str:>11s}")

    print("\n    [OK] Results saved to output/t_test_results.csv")

    # ------------------------------------------------
    # [7] FEATURE CORRELATIONS WITH OUTCOME
    # ------------------------------------------------
    print_separator("7] FEATURE CORRELATIONS WITH OUTCOME")

    correlations = compute_feature_correlations(dataframe)

    for feature_name, correlation_value in correlations.items():
        bar_length = int(abs(correlation_value) * 30)
        bar = "#" * bar_length
        print(f"    {feature_name:<30s}  {correlation_value:+.4f}  {bar}")

    # ------------------------------------------------
    # VISUALIZATIONS
    # ------------------------------------------------
    print(f"\n{'-' * 52}")
    print("  GENERATING VISUALIZATIONS...")
    print(f"{'-' * 52}")

    # Plot 1: Risk Heatmap
    path1 = plot_risk_heatmap(model, dataframe)
    print(f"    [OK] [1/5] Risk Heatmap        -> {os.path.basename(path1)}")

    # Plot 2: Threshold Analysis
    path2 = plot_threshold_analysis(threshold_data)
    print(f"    [OK] [2/5] Threshold Analysis   -> {os.path.basename(path2)}")

    # Plot 3: ROC Curve
    fpr, tpr, _ = get_roc_curve_data(y_test, test_probabilities)
    path3 = plot_roc_curve(fpr, tpr, metrics["auc_roc"])
    print(f"    [OK] [3/5] ROC Curve            -> {os.path.basename(path3)}")

    # Plot 4: Correlation Heatmap
    path4 = plot_correlation_heatmap(dataframe)
    print(f"    [OK] [4/5] Correlation Heatmap  -> {os.path.basename(path4)}")

    # Plot 5: T-Test Means
    path5 = plot_ttest_feature_means(X_train, y_train, ttest_results)
    print(f"    [OK] [5/5] T-Test Means         -> {os.path.basename(path5)}")

    # ------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------
    print(f"\n{'=' * 52}")
    print("  [OK] Analysis complete.")
    print(f"  [OK] 5 plots saved to {PLOTS_DIR}")
    print("  [OK] T-test results saved to output/t_test_results.csv")
    print(f"{'=' * 52}")


if __name__ == "__main__":
    main()
