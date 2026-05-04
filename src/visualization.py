"""
Visualization module: 5 essential diagnostic plots.

Plots:
    1. 2D Risk Heatmap – P(Diabetic) contours for Glucose vs BMI
    2. Threshold Analysis – Sensitivity, Specificity, F1 vs threshold
    3. ROC Curve – Discrimination ability with AUC
    4. Correlation Heatmap – Feature-feature and feature-outcome correlations
    5. T-Test Means – Feature mean values by class with significance indicators

All plots saved as PNG at 150 DPI to output/plots/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
import sys

# Use non-interactive backend to avoid display issues in headless environments
matplotlib.use("Agg")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    FEATURE_NAMES,
    TARGET_COLUMN,
    PLOT_DPI,
    PLOTS_DIR,
    PLOT_FILENAMES,
    HEATMAP_GRID_RESOLUTION,
)


def _ensure_plots_directory() -> None:
    """Create the output/plots/ directory if it does not exist."""
    os.makedirs(PLOTS_DIR, exist_ok=True)


def plot_risk_heatmap(
    model,
    dataframe: pd.DataFrame,
    glucose_index: int = 1,
    bmi_index: int = 5,
) -> str:
    """
    Plot 1: 2D Risk Heatmap of P(Diabetic) for Glucose vs BMI.

    Creates a filled contour plot of posterior probability across a
    Glucose × BMI grid, overlaid with actual patient data points
    colour-coded by true outcome.

    Args:
        model: Fitted BayesDiabetesClassifier instance.
        dataframe: Cleaned DataFrame for extracting data ranges.
        glucose_index: Index of Glucose in the feature array (default 1).
        bmi_index: Index of BMI in the feature array (default 5).

    Returns:
        Path to the saved PNG file.
    """
    _ensure_plots_directory()

    # ── Build evaluation grid ──
    glucose_values = dataframe["Glucose"].values
    bmi_values = dataframe["BMI"].values

    glucose_range = np.linspace(glucose_values.min(), glucose_values.max(), HEATMAP_GRID_RESOLUTION)
    bmi_range = np.linspace(bmi_values.min(), bmi_values.max(), HEATMAP_GRID_RESOLUTION)
    glucose_grid, bmi_grid = np.meshgrid(glucose_range, bmi_range)

    # For each grid point, create an 8D feature vector using class means
    # for all features except Glucose and BMI
    mean_features = dataframe[FEATURE_NAMES].mean().values
    probability_grid = np.zeros_like(glucose_grid)

    for row in range(HEATMAP_GRID_RESOLUTION):
        # Build batch of 8D vectors for this row
        batch = np.tile(mean_features, (HEATMAP_GRID_RESOLUTION, 1))
        batch[:, glucose_index] = glucose_grid[row, :]
        batch[:, bmi_index] = bmi_grid[row, :]
        probability_grid[row, :] = model.predict_probabilities(batch)

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(12, 9))

    # Filled contours
    contourf = ax.contourf(
        glucose_grid, bmi_grid, probability_grid,
        levels=50, cmap="RdYlBu_r", alpha=0.9,
    )
    colorbar = fig.colorbar(contourf, ax=ax, label="P(Diabetic | Glucose, BMI)")

    # Decision boundary contour lines
    contour_lines = ax.contour(
        glucose_grid, bmi_grid, probability_grid,
        levels=[0.2, 0.4, 0.5, 0.6, 0.8],
        colors="black", linewidths=1.5, linestyles="--",
    )
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt="%.1f")

    # Actual patient scatter
    non_diabetic = dataframe[dataframe[TARGET_COLUMN] == 0]
    diabetic = dataframe[dataframe[TARGET_COLUMN] == 1]

    ax.scatter(
        non_diabetic["Glucose"], non_diabetic["BMI"],
        c="dodgerblue", alpha=0.6, s=25, label="Non-diabetic",
        edgecolors="white", linewidth=0.4,
    )
    ax.scatter(
        diabetic["Glucose"], diabetic["BMI"],
        c="crimson", alpha=0.6, s=25, label="Diabetic",
        edgecolors="white", linewidth=0.4,
    )

    ax.set_xlabel("Glucose (log-transformed)", fontsize=13)
    ax.set_ylabel("BMI", fontsize=13)
    ax.set_title("Diabetes Risk Heatmap: P(Diabetic | Glucose, BMI)", fontsize=15)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    filepath = os.path.join(PLOTS_DIR, PLOT_FILENAMES["risk_heatmap"])
    fig.savefig(filepath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_threshold_analysis(threshold_data: dict) -> str:
    """
    Plot 2: Sensitivity, Specificity, and F1 across decision thresholds.

    Shows the Youden's J optimal threshold as a vertical line.

    Args:
        threshold_data: Dictionary from find_optimal_threshold().

    Returns:
        Path to the saved PNG file.
    """
    _ensure_plots_directory()

    thresholds = threshold_data["thresholds"]
    sensitivities = threshold_data["sensitivities"]
    specificities = threshold_data["specificities"]
    f1_scores = threshold_data["f1_scores"]
    optimal_threshold = threshold_data["optimal_threshold"]

    fig, ax = plt.subplots(figsize=(11, 7))

    ax.plot(thresholds, sensitivities, label="Sensitivity (Recall)", linewidth=2.5, color="#E74C3C")
    ax.plot(thresholds, specificities, label="Specificity", linewidth=2.5, color="#2E86C1")
    ax.plot(thresholds, f1_scores, label="F1 Score", linewidth=2.5, color="#27AE60", linestyle="--")

    # Optimal threshold marker
    ax.axvline(
        optimal_threshold, color="#8E44AD", linewidth=2, linestyle=":",
        label=f"Optimal Threshold = {optimal_threshold:.3f}",
    )

    # Mark the intersection point
    optimal_index = np.argmin(np.abs(thresholds - optimal_threshold))
    ax.scatter(
        [optimal_threshold, optimal_threshold],
        [sensitivities[optimal_index], specificities[optimal_index]],
        color="#8E44AD", s=100, zorder=5,
    )

    ax.set_xlabel("Decision Threshold", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Threshold Analysis: Sensitivity vs Specificity Trade-off", fontsize=15)
    ax.legend(fontsize=11, loc="center right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    filepath = os.path.join(PLOTS_DIR, PLOT_FILENAMES["threshold_analysis"])
    fig.savefig(filepath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc_value: float) -> str:
    """
    Plot 3: ROC Curve with AUC score.

    Args:
        fpr: False positive rate array.
        tpr: True positive rate array.
        auc_value: Area under the ROC curve.

    Returns:
        Path to the saved PNG file.
    """
    _ensure_plots_directory()

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.plot(fpr, tpr, color="#E74C3C", linewidth=2.5, label=f"ROC Curve (AUC = {auc_value:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", linewidth=1.5, linestyle="--", label="Random Classifier")

    # Shade the AUC area
    ax.fill_between(fpr, tpr, alpha=0.15, color="#E74C3C")

    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=13)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=15)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()

    filepath = os.path.join(PLOTS_DIR, PLOT_FILENAMES["roc_curve"])
    fig.savefig(filepath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> str:
    """
    Plot 4: Correlation heatmap of all features and the Outcome variable.

    Args:
        dataframe: Cleaned DataFrame with features and Outcome.

    Returns:
        Path to the saved PNG file.
    """
    _ensure_plots_directory()

    columns = FEATURE_NAMES + [TARGET_COLUMN]
    correlation_matrix = dataframe[columns].corr()

    fig, ax = plt.subplots(figsize=(12, 10))

    # Short labels for readability
    short_labels = [
        "Preg", "Gluc", "BP", "Skin", "Ins",
        "BMI", "DPF", "Age", "Outcome",
    ]

    im = ax.imshow(correlation_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, shrink=0.85, label="Pearson Correlation")

    # Annotate cells with correlation values
    for row in range(len(columns)):
        for col in range(len(columns)):
            value = correlation_matrix.values[row, col]
            text_color = "white" if abs(value) > 0.5 else "black"
            ax.text(
                col, row, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                color=text_color, fontweight="bold",
            )

    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(short_labels, fontsize=11, rotation=45, ha="right")
    ax.set_yticklabels(short_labels, fontsize=11)
    ax.set_title("Feature Correlation Heatmap (with Outcome)", fontsize=15)
    fig.tight_layout()

    filepath = os.path.join(PLOTS_DIR, PLOT_FILENAMES["correlation_heatmap"])
    fig.savefig(filepath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath


def plot_ttest_feature_means(
    X_train: np.ndarray,
    y_train: np.ndarray,
    ttest_results: pd.DataFrame,
    feature_names: list = FEATURE_NAMES,
) -> str:
    """
    Plot 5: Grouped bar chart of feature means by class with significance.

    For each feature, shows the mean value for non-diabetic (class 0) and
    diabetic (class 1) patients, with significance stars from t-tests.

    Args:
        X_train: Training features, shape (n, 8).
        y_train: Training labels, shape (n,).
        ttest_results: DataFrame from perform_welch_t_tests().
        feature_names: Names of the 8 features.

    Returns:
        Path to the saved PNG file.
    """
    _ensure_plots_directory()

    means_class0 = X_train[y_train == 0].mean(axis=0)
    means_class1 = X_train[y_train == 1].mean(axis=0)

    short_labels = ["Preg", "Gluc", "BP", "Skin", "Ins", "BMI", "DPF", "Age"]

    x_positions = np.arange(len(feature_names))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(14, 8))

    bars0 = ax.bar(
        x_positions - bar_width / 2, means_class0,
        bar_width, label="Non-Diabetic (0)", color="#2E86C1", alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )
    bars1 = ax.bar(
        x_positions + bar_width / 2, means_class1,
        bar_width, label="Diabetic (1)", color="#E74C3C", alpha=0.85,
        edgecolor="white", linewidth=0.8,
    )

    # Compute the global max height to set consistent y-axis padding
    global_max = max(means_class0.max(), means_class1.max())

    # Add significance stars and p-value annotations above each pair of bars
    for index in range(len(feature_names)):
        max_height = max(means_class0[index], means_class1[index])
        stars = ttest_results.iloc[index]["Stars"]
        p_val = ttest_results.iloc[index]["p_value"]

        if stars:
            # Format p-value for display
            p_text = "p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"

            # Place star + p-value above the taller bar
            ax.text(
                x_positions[index], max_height + global_max * 0.03,
                f"{stars}\n({p_text})",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color="#8E44AD",
            )

    # Add headroom so annotations never overlap the title
    ax.set_ylim(0, global_max * 1.25)

    ax.set_xlabel("Features", fontsize=13)
    ax.set_ylabel("Mean Value", fontsize=13)
    ax.set_title("Feature Means by Outcome Class (with T-Test Significance)", fontsize=15, pad=15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_labels, fontsize=11)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    # Footnote explaining significance stars
    fig.text(
        0.5, 0.01,
        "Significance: * p<0.05,  ** p<0.01,  *** p<0.001  (Welch's t-test)",
        ha="center", fontsize=10, fontstyle="italic", color="#555555",
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])  # Leave room for footnote

    filepath = os.path.join(PLOTS_DIR, PLOT_FILENAMES["ttest_means"])
    fig.savefig(filepath, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return filepath
