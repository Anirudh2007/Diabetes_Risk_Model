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

# Output directory
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

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
# Decision Threshold
# ──────────────────────────────────────────────

# Fixed decision threshold used for evaluation and web UI
DECISION_THRESHOLD = 0.5
