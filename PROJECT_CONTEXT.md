# PROJECT CONTEXT: 8D Bayesian Diabetes Risk Prediction
# Copy-paste this ENTIRE file as your first message to any AI assistant.
# Then upload your code files as attachments.

## WHAT THIS PROJECT IS
An 8-dimensional Bayesian classifier for diabetes risk prediction using the UCI Pima Indians Diabetes Dataset (768 samples, 8 features, binary outcome).

## PROJECT STRUCTURE
```
Project/
├── config.py                  # Centralized paths, parameters, constants
├── main.py                    # Single entry point - runs full pipeline
├── requirements.txt           # numpy, pandas, scipy, scikit-learn, matplotlib
├── archive/
│   └── diabetes.csv           # UCI Pima Indians dataset (768 rows, 9 columns)
├── src/
│   ├── __init__.py            # Package init
│   ├── data_cleaning.py       # Loading, Gaussian imputation, log-transform, stratified split
│   ├── model.py               # BayesDiabetesClassifier (8D multivariate normal Bayes)
│   ├── evaluation.py          # Youden's J threshold, metrics, Welch's t-tests, correlations
│   └── visualization.py       # 5 plots: heatmap, threshold, ROC, correlation, t-test means
└── output/
    ├── t_test_results.csv     # Welch's t-test results for all 8 features
    └── plots/                 # 5 PNG diagnostic plots at 150 DPI
```

## DATASET COLUMNS
| Column | Type | Description |
|--------|------|-------------|
| Pregnancies | int | Number of pregnancies |
| Glucose | int | Plasma glucose (2h OGTT) |
| BloodPressure | int | Diastolic BP (mm Hg) |
| SkinThickness | int | Triceps skinfold (mm) |
| Insulin | int | 2-hour serum insulin (uU/ml) |
| BMI | float | Body mass index (kg/m^2) |
| DiabetesPedigreeFunction | float | Genetic diabetes likelihood |
| Age | int | Age in years |
| Outcome | int | 0 = non-diabetic, 1 = diabetic |

Class distribution: 500 non-diabetic (65.1%), 268 diabetic (34.9%)

## DATA CLEANING PIPELINE
1. Replace biologically impossible zeros in [Glucose, BloodPressure, SkinThickness, Insulin, BMI] with Gaussian imputation: value = column_mean + N(0, 0.1)
2. Log-transform skewed features: Pregnancies, Insulin -> log(x + 1)
3. Validate: no NaN, no Inf, Outcome in {0,1}
4. Stratified 70/30 train-test split (random_state=42)

## MODEL ARCHITECTURE
- Class-conditional multivariate normal distributions: N(mu_k, Sigma_k) for k in {0, 1}
- Covariance regularization: Sigma' = Sigma + 1e-6 * I (positive definiteness)
- Posterior via Bayes' theorem: P(k|x) = P(x|k)*P(k) / sum_j[P(x|j)*P(j)]
- Edge case handling: zero-denominator fallback to prior, NaN/Inf clipping

## CURRENT MODEL RESULTS (Test Set, n=231)
- Optimal threshold (Youden's J): 0.141
- Accuracy: 69.3%
- Sensitivity: 92.6% (75/81 diabetics caught)
- Specificity: 56.7%
- Precision: 53.6%
- F1 Score: 0.6787
- AUC-ROC: 0.7864
- Confusion Matrix: TN=85, FP=65, FN=6, TP=75

## STATISTICAL VALIDATION
All 8 features significant via Welch's t-test (p < 0.05):
- Glucose (p<0.0001), BMI (p<0.0001), Age (p<0.0001): strongest discriminators
- DiabetesPedigreeFunction (p=0.0013): weakest but still significant

## FEATURE CORRELATIONS WITH OUTCOME (sorted)
Glucose: +0.49, BMI: +0.31, Insulin: +0.25, Age: +0.24,
SkinThickness: +0.22, Pregnancies: +0.18, DPF: +0.17, BloodPressure: +0.17

## TECH STACK
Python 3.10+, numpy, pandas, scipy, scikit-learn, matplotlib
No deep learning. No black-box ML. Pure Bayesian probability.

## KEY DESIGN DECISIONS
1. Low threshold (0.141) prioritizes sensitivity over specificity (medical screening)
2. Gaussian noise imputation preserves variance (vs. mean-only imputation)
3. Log-transform on Pregnancies/Insulin improves Gaussian assumption fit
4. Regularization prevents singular covariance matrices on small sample sizes
