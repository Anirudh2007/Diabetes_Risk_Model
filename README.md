# 8D Bayesian Diabetes Risk Prediction Model

A robust, full-dimensional Bayesian diagnostic model for reliable diabetes risk
prediction using the complete **UCI Pima Indians Diabetes Dataset** (768 samples,
8 features).

---

## Project Structure

```
Project/
├── config.py                   # Paths, parameters, constants
├── main.py                     # Single execution entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py        # Data loading, cleaning, splitting
│   ├── model.py                # 8D Bayes classifier
│   ├── evaluation.py           # Metrics, tests, threshold optimization
│   └── visualization.py        # 5 essential diagnostic plots
├── archive/
│   └── diabetes.csv            # Raw dataset (UCI Pima Indians)
└── output/
    ├── plots/                  # 5 saved PNG plots (150 DPI)
    │   ├── 01_risk_heatmap_glucose_bmi.png
    │   ├── 02_threshold_analysis.png
    │   ├── 03_roc_curve.png
    │   ├── 04_correlation_heatmap.png
    │   └── 05_ttest_feature_means.png
    └── t_test_results.csv      # Welch's t-test statistical results
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the model

```bash
python main.py
```

All output (metrics, plots, CSV) is generated automatically.

---

## Features Used (8D)

| # | Feature                    | Description                              |
|---|----------------------------|------------------------------------------|
| 1 | Pregnancies                | Number of times pregnant                 |
| 2 | Glucose                    | Plasma glucose concentration (2h OGTT)   |
| 3 | BloodPressure              | Diastolic blood pressure (mm Hg)         |
| 4 | SkinThickness              | Triceps skin fold thickness (mm)         |
| 5 | Insulin                    | 2-Hour serum insulin (μU/ml)             |
| 6 | BMI                        | Body mass index (kg/m²)                  |
| 7 | DiabetesPedigreeFunction   | Diabetes pedigree function               |
| 8 | Age                        | Age in years                             |

---

## Methodology

### Data Cleaning
- **Invalid zeros**: Glucose, BloodPressure, SkinThickness, Insulin, and BMI
  zeros are replaced using Gaussian noise imputation: `value = μ + ε`,
  where `ε ~ N(0, 0.1)`.
- **Log transform**: Pregnancies and Insulin are log-transformed for normality.
- **Validation**: Checks for NaN, Inf, and out-of-range values.

### Model
- **Multivariate normal** class-conditional distributions N(μ, Σ) fitted
  per class on the 8D feature space.
- **Bayes' theorem** computes posterior P(Diabetic | x).
- **Regularization**: `Σ' = Σ + 1e-6 · I` ensures positive definiteness.

### Threshold Optimization
- **Youden's J statistic** (`J = Sensitivity + Specificity − 1`) finds
  the data-driven optimal decision threshold (not an arbitrary 0.5).

### Statistical Validation
- **Welch's t-tests** on all 8 features (α = 0.05).
- **Pearson correlations** with the Outcome variable.
- **Eigenvalue analysis** verifies covariance matrix integrity.

---

## Output Interpretation

| Metric      | Meaning                                          |
|-------------|--------------------------------------------------|
| Accuracy    | Overall correctness of predictions               |
| Sensitivity | Proportion of actual diabetics correctly detected |
| Specificity | Proportion of actual non-diabetics correctly identified |
| Precision   | When model predicts diabetic, how often correct  |
| F1 Score    | Harmonic mean of precision and sensitivity       |
| AUC-ROC     | Overall discrimination ability (1.0 = perfect)   |

### Plots
1. **Risk Heatmap** – Probability contours on Glucose × BMI plane
2. **Threshold Analysis** – How threshold affects Sens/Spec/F1
3. **ROC Curve** – Model discrimination with AUC value
4. **Correlation Heatmap** – Feature inter-relationships
5. **T-Test Means** – Mean differences by class with significance

---

## Dataset

UCI Pima Indians Diabetes Dataset  
Source: [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
768 female patients of Pima Indian heritage, ≥ 21 years of age.

---

## License

Academic use for DSC 10 coursework.
