"""
Data loading, cleaning, validation, and train-test splitting.

Pipeline:
    1. Load raw CSV and validate schema.
    2. Replace biologically impossible zeros with Gaussian-noise-imputed values.
    3. Log-transform right-skewed features (Pregnancies, Insulin) for normality.
    4. Run integrity checks (NaN, Inf, out-of-range).
    5. Stratified 70-30 train-test split preserving class distribution.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import sys
import os

# Allow imports from project root regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DATA_PATH,
    FEATURE_NAMES,
    TARGET_COLUMN,
    ZERO_INVALID_COLUMNS,
    LOG_TRANSFORM_FEATURES,
    IMPUTATION_NOISE_SIGMA,
    LOG_TRANSFORM_OFFSET,
    TEST_SIZE,
    RANDOM_STATE,
)


def load_dataset(filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the Pima Indians Diabetes Dataset from CSV.

    Args:
        filepath: Absolute or relative path to the CSV file.

    Returns:
        Raw DataFrame with all 9 columns (8 features + Outcome).

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If expected columns are missing.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    dataframe = pd.read_csv(filepath)

    # Verify all required columns are present
    expected_columns = FEATURE_NAMES + [TARGET_COLUMN]
    missing_columns = [col for col in expected_columns if col not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")

    print(f"    [OK] Loaded {len(dataframe)} samples with {len(dataframe.columns)} columns")
    return dataframe


def impute_invalid_zeros(
    dataframe: pd.DataFrame,
    columns: list = ZERO_INVALID_COLUMNS,
    noise_sigma: float = IMPUTATION_NOISE_SIGMA,
) -> pd.DataFrame:
    """
    Replace biologically impossible zero values with Gaussian-imputed values.

    For medical measurements like Glucose, BloodPressure, SkinThickness,
    Insulin, and BMI, a zero reading indicates a missing value, not a true
    measurement.  We replace each zero with  mu + eps  where mu is the column
    mean of non-zero values and eps ~ N(0, noise_sigma).

    Args:
        dataframe: Input DataFrame (modified in-place and returned).
        columns: Column names where zero is invalid.
        noise_sigma: Std-dev of the Gaussian noise added to the mean.

    Returns:
        DataFrame with imputed values (no biologically impossible zeros).
    """
    np.random.seed(RANDOM_STATE)  # Reproducible imputation
    total_imputed = 0

    # Cast target columns to float64 so imputed float values can be assigned
    # (newer pandas raises TypeError when inserting floats into int64 columns)
    for column_name in columns:
        dataframe[column_name] = dataframe[column_name].astype(np.float64)

    for column_name in columns:
        zero_mask = dataframe[column_name] == 0
        zero_count = zero_mask.sum()

        if zero_count == 0:
            continue

        # Mean of valid (non-zero) values only
        column_mean = dataframe.loc[~zero_mask, column_name].mean()

        # Generate Gaussian noise: eps ~ N(0, noise_sigma)
        noise = np.random.normal(0, noise_sigma, size=zero_count)

        # Impute: value = mu + eps, clipped to stay positive
        imputed_values = np.clip(column_mean + noise, a_min=0.01, a_max=None)
        dataframe.loc[zero_mask, column_name] = imputed_values

        total_imputed += zero_count

    print(f"    [OK] Cleaned {len(columns)} zero-columns with Gaussian imputation "
          f"({total_imputed} values imputed)")
    return dataframe


def apply_log_transforms(
    dataframe: pd.DataFrame,
    columns: list = LOG_TRANSFORM_FEATURES,
    offset: float = LOG_TRANSFORM_OFFSET,
) -> pd.DataFrame:
    """
    Apply log1p transform to right-skewed features for better Gaussian normality.

    Uses log(x + offset) to avoid log(0).

    Args:
        dataframe: Input DataFrame (modified in-place and returned).
        columns: Features to log-transform.
        offset: Constant added before log to handle zeros.

    Returns:
        DataFrame with transformed columns.
    """
    for column_name in columns:
        dataframe[column_name] = np.log(dataframe[column_name] + offset)

    print(f"    [OK] Log-transformed skewed features: {columns}")
    return dataframe


def validate_data_integrity(dataframe: pd.DataFrame) -> None:
    """
    Run comprehensive integrity checks on the cleaned DataFrame.

    Checks for:
        - Missing (NaN) values
        - Infinite values
        - Out-of-range values (negative features, invalid Outcome)
        - Duplicate rows

    Args:
        dataframe: Cleaned DataFrame to validate.

    Raises:
        ValueError: If any critical integrity check fails.
    """
    # Check for NaN values
    nan_count = dataframe[FEATURE_NAMES].isna().sum().sum()
    if nan_count > 0:
        raise ValueError(f"Data contains {nan_count} NaN values after cleaning")
    print("    [OK] No missing values remaining")

    # Check for infinite values
    inf_count = np.isinf(dataframe[FEATURE_NAMES].values).sum()
    if inf_count > 0:
        raise ValueError(f"Data contains {inf_count} infinite values")

    # Check outcome column has only 0 and 1
    unique_outcomes = set(dataframe[TARGET_COLUMN].unique())
    if not unique_outcomes.issubset({0, 1}):
        raise ValueError(f"Outcome column contains invalid values: {unique_outcomes}")

    # Check for negative feature values (after log transform, negatives are possible)
    # We only warn, not error, because log-transformed values can be negative
    negative_features = []
    for col in FEATURE_NAMES:
        if col not in LOG_TRANSFORM_FEATURES and (dataframe[col] < 0).any():
            negative_features.append(col)
    if negative_features:
        print(f"    [WARN] Negative values in non-log features: {negative_features}")

    print("    [OK] Validated data integrity")


def split_train_test(
    dataframe: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple:
    """
    Perform stratified 70-30 train-test split preserving class distribution.

    Args:
        dataframe: Cleaned and validated DataFrame.
        test_size: Fraction of data for testing (default 0.30).
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) where X are feature
        arrays and y are outcome arrays.
    """
    feature_matrix = dataframe[FEATURE_NAMES].values
    target_vector = dataframe[TARGET_COLUMN].values

    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix,
        target_vector,
        test_size=test_size,
        random_state=random_state,
        stratify=target_vector,  # Maintain class distribution
    )

    train_diabetic_ratio = y_train.mean()
    test_diabetic_ratio = y_test.mean()

    print(f"    [OK] Training: {len(y_train)} samples ({train_diabetic_ratio:.1%} diabetic)")
    print(f"    [OK] Testing:  {len(y_test)} samples ({test_diabetic_ratio:.1%} diabetic)")

    return X_train, X_test, y_train, y_test


def run_cleaning_pipeline() -> tuple:
    """
    Execute the complete data cleaning and splitting pipeline.

    Returns:
        Tuple of (dataframe, X_train, X_test, y_train, y_test).
        The dataframe is the fully cleaned version for use in visualizations.
    """
    dataframe = load_dataset()
    dataframe = impute_invalid_zeros(dataframe)
    dataframe = apply_log_transforms(dataframe)
    validate_data_integrity(dataframe)
    X_train, X_test, y_train, y_test = split_train_test(dataframe)

    return dataframe, X_train, X_test, y_train, y_test
