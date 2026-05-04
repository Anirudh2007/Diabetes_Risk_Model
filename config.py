"""
Configuration module for the 8D Bayesian Diabetes Risk Prediction model.

Centralizes all paths, parameters, and constants used across the project.
Keeps configuration separate from logic for maintainability.
"""

import os

# ──────────────────────────────────────────────
# Path Configuration
# ──────────────────────────────────────────────

# Base directory is the folder containing this config file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Input data path
DATA_PATH = os.path.join(BASE_DIR, "archive", "diabetes.csv")

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# ──────────────────────────────────────────────
# Dataset Feature Configuration
# ──────────────────────────────────────────────

# All 8 predictor features used in the model
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Target variable
TARGET_COLUMN = "Outcome"

# Columns where zero is biologically impossible (indicates missing data)
ZERO_INVALID_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

# Features to log-transform for Gaussian normality (right-skewed distributions)
LOG_TRANSFORM_FEATURES = [
    "Pregnancies",
    "Insulin",
]

# ──────────────────────────────────────────────
# Data Cleaning Parameters
# ──────────────────────────────────────────────

# Standard deviation for Gaussian noise imputation (μ ± ε)
IMPUTATION_NOISE_SIGMA = 0.1

# Small constant added before log-transform to avoid log(0)
LOG_TRANSFORM_OFFSET = 1.0

# ──────────────────────────────────────────────
# Train-Test Split Parameters
# ──────────────────────────────────────────────

# Fraction of data reserved for testing
TEST_SIZE = 0.30

# Random seed for reproducibility
RANDOM_STATE = 42

# ──────────────────────────────────────────────
# Model Parameters
# ──────────────────────────────────────────────

# Regularization added to covariance matrix diagonal for positive definiteness
COVARIANCE_REGULARIZATION = 1e-6

# ──────────────────────────────────────────────
# Threshold Optimization
# ──────────────────────────────────────────────

# Number of candidate thresholds to evaluate between 0 and 1
THRESHOLD_SEARCH_POINTS = 1000

# ──────────────────────────────────────────────
# Statistical Testing
# ──────────────────────────────────────────────

# Significance level for Welch's t-tests
SIGNIFICANCE_ALPHA = 0.05

# ──────────────────────────────────────────────
# Visualization Parameters
# ──────────────────────────────────────────────

# Resolution for saved plots
PLOT_DPI = 150

# Grid resolution for the 2D risk heatmap
HEATMAP_GRID_RESOLUTION = 100

# Plot filenames
PLOT_FILENAMES = {
    "risk_heatmap": "01_risk_heatmap_glucose_bmi.png",
    "threshold_analysis": "02_threshold_analysis.png",
    "roc_curve": "03_roc_curve.png",
    "correlation_heatmap": "04_correlation_heatmap.png",
    "ttest_means": "05_ttest_feature_means.png",
}
