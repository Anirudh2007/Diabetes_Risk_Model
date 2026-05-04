"""
8-Dimensional Bayesian Classifier for diabetes risk prediction.

Implements a Naive Bayes classifier with multivariate normal class-conditional
densities.  Each class (diabetic / non-diabetic) is modelled as:

    P(x | class) = N(x; μ_class, Σ_class)

where x ∈ ℝ⁸ is the full feature vector.

Posterior probability via Bayes' theorem:

    P(class=1 | x) = P(x | class=1) · P(class=1)
                      ─────────────────────────────
                            P(x)

with  P(x) = Σ_k  P(x | class=k) · P(class=k).
"""

import numpy as np
from scipy.stats import multivariate_normal

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COVARIANCE_REGULARIZATION, FEATURE_NAMES


class BayesDiabetesClassifier:
    """
    Full 8-dimensional Gaussian Bayes classifier.

    Attributes:
        mean_non_diabetic: Mean vector μ₀ ∈ ℝ⁸ for class 0.
        mean_diabetic: Mean vector μ₁ ∈ ℝ⁸ for class 1.
        covariance_non_diabetic: Covariance Σ₀ ∈ ℝ⁸ˣ⁸ for class 0.
        covariance_diabetic: Covariance Σ₁ ∈ ℝ⁸ˣ⁸ for class 1.
        prior_non_diabetic: Prior P(class=0).
        prior_diabetic: Prior P(class=1).
        distribution_non_diabetic: Fitted scipy multivariate_normal for class 0.
        distribution_diabetic: Fitted scipy multivariate_normal for class 1.
        feature_names: List of 8 feature names.
        is_fitted: Whether the model has been trained.
    """

    def __init__(
        self,
        regularization: float = COVARIANCE_REGULARIZATION,
        feature_names: list = None,
    ):
        """
        Initialize the classifier.

        Args:
            regularization: Value added to covariance diagonals for
                            positive definiteness (default 1e-6).
            feature_names: Optional list of feature names for reporting.
        """
        self.regularization = regularization
        self.feature_names = feature_names or FEATURE_NAMES

        # Model parameters — populated during fit()
        self.mean_non_diabetic = None
        self.mean_diabetic = None
        self.covariance_non_diabetic = None
        self.covariance_diabetic = None
        self.prior_non_diabetic = None
        self.prior_diabetic = None
        self.distribution_non_diabetic = None
        self.distribution_diabetic = None
        self.is_fitted = False

        # Eigenvalue diagnostics
        self.min_eigenvalue_class0 = None
        self.min_eigenvalue_class1 = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "BayesDiabetesClassifier":
        """
        Train the 8D Bayes model on training data only.

        Estimates class-conditional multivariate normal parameters
        (mean vectors and covariance matrices) plus class priors.

        Args:
            X_train: Training feature matrix of shape (n_samples, 8).
            y_train: Training labels of shape (n_samples,), values in {0, 1}.

        Returns:
            self (for method chaining).

        Raises:
            ValueError: If feature dimensions mismatch or labels are invalid.
        """
        # ── Input validation ──
        if X_train.shape[1] != 8:
            raise ValueError(
                f"Expected 8 features, got {X_train.shape[1]}. "
                f"Provide all features: {self.feature_names}"
            )

        unique_labels = set(np.unique(y_train))
        if not unique_labels.issubset({0, 1}):
            raise ValueError(f"Labels must be 0 or 1, got: {unique_labels}")

        # ── Separate classes ──
        X_class0 = X_train[y_train == 0]  # Non-diabetic samples
        X_class1 = X_train[y_train == 1]  # Diabetic samples

        # ── Prior probabilities from training frequencies ──
        self.prior_non_diabetic = len(X_class0) / len(X_train)
        self.prior_diabetic = len(X_class1) / len(X_train)

        # ── Mean vectors ──
        self.mean_non_diabetic = np.mean(X_class0, axis=0)
        self.mean_diabetic = np.mean(X_class1, axis=0)

        # ── Covariance matrices with regularization ──
        self.covariance_non_diabetic = self._compute_regularized_covariance(X_class0)
        self.covariance_diabetic = self._compute_regularized_covariance(X_class1)

        # ── Verify positive definiteness via eigenvalue analysis ──
        self.min_eigenvalue_class0 = self._verify_positive_definite(
            self.covariance_non_diabetic, "Non-diabetic"
        )
        self.min_eigenvalue_class1 = self._verify_positive_definite(
            self.covariance_diabetic, "Diabetic"
        )

        # ── Build multivariate normal distributions ──
        self.distribution_non_diabetic = multivariate_normal(
            mean=self.mean_non_diabetic,
            cov=self.covariance_non_diabetic,
            allow_singular=False,
        )
        self.distribution_diabetic = multivariate_normal(
            mean=self.mean_diabetic,
            cov=self.covariance_diabetic,
            allow_singular=False,
        )

        self.is_fitted = True
        return self

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute P(Diabetic | x) for each sample using Bayes' theorem.

        For a feature vector x:
            P(class=1 | x) = P(x|1)·P(1) / [P(x|0)·P(0) + P(x|1)·P(1)]

        Uses log probabilities to avoid numerical underflow in high dimensions.

        Args:
            X: Feature matrix of shape (n_samples, 8).

        Returns:
            Array of posterior probabilities ∈ [0, 1], shape (n_samples,).

        Raises:
            RuntimeError: If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predicting. Call fit() first.")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # ── Class-conditional log-likelihoods ──
        # log P(x | class=0) and log P(x | class=1)
        log_likelihood_class0 = self.distribution_non_diabetic.logpdf(X)
        log_likelihood_class1 = self.distribution_diabetic.logpdf(X)

        # ── Log numerator and denominator of Bayes' theorem ──
        # log(P(x|1)·P(1)) = log P(x|1) + log P(1)
        log_numerator = log_likelihood_class1 + np.log(self.prior_diabetic)
        
        # log(P(x|0)·P(0) + P(x|1)·P(1)) = log_sum_exp(log(P(x|0)·P(0)), log(P(x|1)·P(1)))
        log_term0 = log_likelihood_class0 + np.log(self.prior_non_diabetic)
        log_term1 = log_likelihood_class1 + np.log(self.prior_diabetic)
        log_denominator = np.logaddexp(log_term0, log_term1)

        # ── Log posterior probability ──
        # log P(class=1 | x) = log(P(x|1)·P(1)) - log(denominator)
        log_posterior = log_numerator - log_denominator

        # ── Convert back to probability space ──
        probabilities = np.exp(log_posterior)

        # ── Final NaN/Inf safety check ──
        probabilities = np.nan_to_num(
            probabilities,
            nan=self.prior_diabetic,
            posinf=1.0,
            neginf=0.0,
        )
        probabilities = np.clip(probabilities, 0.0, 1.0)

        return np.atleast_1d(probabilities)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict binary class labels using a decision threshold.

        Args:
            X: Feature matrix of shape (n_samples, 8).
            threshold: Decision boundary; predict 1 if P(diabetic) ≥ threshold.

        Returns:
            Binary predictions array of shape (n_samples,).
        """
        probabilities = self.predict_probabilities(X)
        return (probabilities >= threshold).astype(int)

    def _compute_regularized_covariance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the sample covariance matrix with diagonal regularization.

        Adds  λ·I  (where λ = self.regularization) to the diagonal to
        guarantee positive definiteness, preventing singular matrix errors
        in the multivariate normal PDF.

        Args:
            X: Data matrix of shape (n_samples, 8).

        Returns:
            Regularized covariance matrix of shape (8, 8).
        """
        covariance = np.cov(X, rowvar=False)

        # Regularization: Σ' = Σ + λI
        covariance += np.eye(covariance.shape[0]) * self.regularization

        return covariance

    def _verify_positive_definite(
        self, covariance_matrix: np.ndarray, class_label: str
    ) -> float:
        """
        Verify covariance matrix is positive definite via eigenvalue analysis.

        A symmetric matrix is positive definite iff all eigenvalues > 0.

        Args:
            covariance_matrix: Covariance matrix to verify.
            class_label: Human-readable class name for diagnostics.

        Returns:
            Minimum eigenvalue (should be > 0).

        Raises:
            ValueError: If the matrix is not positive definite even after
                        regularization.
        """
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        min_eigenvalue = np.min(eigenvalues)

        if min_eigenvalue <= 0:
            raise ValueError(
                f"Covariance matrix for '{class_label}' is NOT positive definite. "
                f"Min eigenvalue = {min_eigenvalue:.2e}. "
                f"Increase regularization parameter."
            )

        return min_eigenvalue

    def get_model_summary(self) -> dict:
        """
        Return a dictionary of trained model parameters for reporting.

        Returns:
            Dictionary with means, covariances, priors, and eigenvalue info.
        """
        if not self.is_fitted:
            return {"error": "Model not fitted"}

        return {
            "prior_diabetic": self.prior_diabetic,
            "prior_non_diabetic": self.prior_non_diabetic,
            "mean_non_diabetic": self.mean_non_diabetic,
            "mean_diabetic": self.mean_diabetic,
            "covariance_non_diabetic": self.covariance_non_diabetic,
            "covariance_diabetic": self.covariance_diabetic,
            "min_eigenvalue_class0": self.min_eigenvalue_class0,
            "min_eigenvalue_class1": self.min_eigenvalue_class1,
            "n_features": 8,
            "feature_names": self.feature_names,
        }
