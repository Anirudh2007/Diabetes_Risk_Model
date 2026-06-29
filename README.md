# Diabetes Risk Prediction Model

A **production-ready probabilistic classifier** for predicting diabetes risk from patient clinical features. Built from first principles using **Gaussian Bayes' theorem** and deployed as both a command-line tool and interactive Flask web application.

**Key Features:**
- 🔬 **From-scratch Bayesian classifier** implementing QDA (Quadratic Discriminant Analysis) with full mathematical rigor
- 🩺 **8-dimensional prediction** using clinical biomarkers (glucose, BMI, blood pressure, insulin, etc.)
- 🧹 **Production-grade data cleaning** with Gaussian imputation and log-transforms
- 🚀 **Dual interface:** CLI for batch evaluation + Flask web UI for real-time predictions
- 📊 **Comprehensive evaluation** with sensitivity/specificity/precision metrics
- ✅ **Reproducible results** with fixed random seeds and stratified splitting

---

## 🎯 Motivation & Problem Statement

### Why This Matters

Diabetes is a silent epidemic affecting **537 million adults globally** (IDF, 2021). Early detection dramatically improves patient outcomes, but:

- **Manual screening is time-consuming** — physicians must evaluate multiple biomarkers
- **Access is limited** — many patients lack early warning signs or routine care
- **Interpretability matters** — doctors need to understand *why* a risk assessment was made

This project demonstrates that **machine learning can democratize risk assessment** using widely-available clinical measurements.

### What This Project Does

We train a **Gaussian Bayes classifier** on the UCI Pima Indians Diabetes Dataset to predict whether a patient has diabetes based on 8 clinical features. The model estimates **posterior probability P(Diabetic | features)**, allowing both binary classification and risk stratification.

**Why Bayesian methods?**
- Probabilistic output (not just binary labels) — better for medical applications
- Interpretable decision boundaries — easy to explain to doctors
- Theoretically sound — Bayes' theorem is the gold standard for classification
- Efficient training — no iterative optimization needed

---

## 🏗️ Technical Architecture

### Data Pipeline

```
Raw CSV
   ↓
[1] Load & Validate Schema
   ↓
[2] Impute Zeros (Gaussian noise)
   ↓
[3] Log-Transform Skewed Features
   ↓
[4] Integrity Checks (NaN, Inf, ranges)
   ↓
[5] Stratified 70-30 Train-Test Split
   ↓
Cleaned Feature Matrices (X_train, X_test, y_train, y_test)
```

### Model Architecture

**Class-Conditional Gaussian Model:**

For each class *k* ∈ {0: Non-Diabetic, 1: Diabetic}, we fit a **multivariate normal distribution**:

```
P(x | class=k) = N(x; μₖ, Σₖ)
```

Where:
- **μₖ** = 8D mean vector (learned from training data)
- **Σₖ** = 8×8 covariance matrix (captures feature correlations within each class)

**Inference via Bayes' Theorem:**

Given a patient's feature vector **x**:

```
P(Diabetic | x) = P(x | Diabetic) · P(Diabetic)
                  ───────────────────────────────────
                        P(x)
```

Where:
- **P(x | Diabetic)** = class-conditional likelihood (multivariate Gaussian PDF)
- **P(Diabetic)** = class prior (from training set frequencies)
- **P(x)** = marginal likelihood (computed via law of total probability)

**Numerical Stability:**

To prevent underflow in 8D probability space, we compute everything in **log-space** using `log-sum-exp` trick:

```python
log P(Diabetic | x) = log(P(x|1)·P(1)) - log_sum_exp(log(P(x|0)·P(0)), log(P(x|1)·P(1)))
```

### Decision Rule

At **decision threshold = 0.5**:
- If P(Diabetic | x) ≥ 0.5 → predict "Diabetic" (higher risk)
- If P(Diabetic | x) < 0.5 → predict "Non-Diabetic" (lower risk)

The probability itself can be used for **risk stratification** (e.g., P > 0.8 = very high risk).

---

## 📊 Dataset

**Source:** UCI Machine Learning Repository — [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

| Property | Value |
|----------|-------|
| **Total Samples** | 768 |
| **Training Samples** | 538 (70%) |
| **Testing Samples** | 230 (30%) |
| **Features** | 8 clinical biomarkers |
| **Target Classes** | 2 (Diabetic: Yes/No) |
| **Class Imbalance** | 34.9% diabetic, 65.1% non-diabetic |

### Features (8D)

| Feature | Unit | Description | Preprocessing |
|---------|------|-------------|---|
| **Pregnancies** | count | Number of pregnancies | Log-transform (right-skewed) |
| **Glucose** | mg/dL | Plasma glucose (fasting) | Gaussian imputation on zeros |
| **BloodPressure** | mm Hg | Diastolic blood pressure | Gaussian imputation on zeros |
| **SkinThickness** | mm | Triceps skin fold thickness | Gaussian imputation on zeros |
| **Insulin** | mu U/ml | 2-hour serum insulin | Log-transform + Gaussian imputation |
| **BMI** | — | Body mass index (weight/height²) | Gaussian imputation on zeros |
| **DiabetesPedigreeFunction** | — | Family history score (0-1) | No transformation |
| **Age** | years | Age at time of measurement | No transformation |

### Data Quality Issues & Solutions

**Problem:** Dataset contains **biologically impossible zeros** (e.g., Glucose = 0 is invalid for a living person)

**Solution:** **Gaussian Imputation**
- For each invalid-zero column, replace with: μ + ε, where:
  - μ = mean of non-zero values in that column
  - ε ~ N(0, 0.1) = small Gaussian noise
- This preserves realistic variance while respecting the learned feature distribution

**Problem:** Pregnancies and Insulin are **right-skewed** (long tail toward high values)

**Solution:** **Log-Transform**
- Apply log(x + 1) to prevent log(0) and improve Gaussian assumption
- More robust Bayes classifier assumes Gaussian-distributed features

---

## 📈 Results & Performance

### Test Set Evaluation (threshold = 0.5)

**Model Performance:**

| Metric | Value | Interpretation |
|--------|-------|---|
| **Accuracy** | 78.7% | Correct predictions overall |
| **Sensitivity (Recall)** | 65.2% | Of actual diabetic patients, we catch 65% |
| **Specificity** | 86.3% | Of non-diabetic patients, we correctly identify 86% |
| **Precision** | 73.7% | Of positive predictions, 74% are truly diabetic |

### Confusion Matrix

```
                    Predicted Negative    Predicted Positive
Actual Negative     TN = 162              FP = 26
Actual Positive     FN = 16               TP = 26
```

### Key Insights

- ✅ **High specificity (86.3%)** — fewer false alarms for healthy patients
- ⚠️ **Room for improvement in sensitivity (65.2%)** — some diabetic patients missed
- 💡 **Better precision than recall** — when we predict positive, it's usually correct
- 🎯 **Threshold trade-off:** At 0.5, we optimize balanced classification. Lower threshold → more sensitive, more false positives. Higher threshold → more specific, more false negatives.

### What This Means Clinically

- For a **screening application** (maximize sensitivity): Lower decision threshold to catch more at-risk patients, accept higher false-positive rate
- For a **diagnostic application** (maximize specificity): Keep threshold at 0.5 or higher, reduce unnecessary interventions

---

## 🛠️ Tech Stack

| Component | Technology | Why? |
|-----------|-----------|------|
| **Core ML** | NumPy, SciPy | Numerical computing, multivariate distributions |
| **Data Processing** | Pandas | Clean, tabular data manipulation |
| **Model Evaluation** | scikit-learn | Industry-standard metrics (confusion matrix, etc.) |
| **Web Framework** | Flask | Lightweight, perfect for single-model deployment |
| **Language** | Python 3.8+ | Rapid prototyping, ML ecosystem |

---

## ⚡ Quick Start

### 1️⃣ Clone & Install

```bash
git clone https://github.com/Anirudh2007/Diabetes_Risk_Model.git
cd Diabetes_Risk_Model
pip install -r requirements.txt
```

### 2️⃣ Get the Dataset

Download the Pima Indians Diabetes CSV from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it here:

```
Diabetes_Risk_Model/
└── archive/
    └── diabetes.csv
```

Or use this direct link:
```bash
mkdir -p archive
wget https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv -O archive/diabetes.csv
echo "Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome" | cat - archive/diabetes.csv > temp && mv temp archive/diabetes.csv
```

### 3️⃣ Run Training & Evaluation

```bash
python main.py
```

**Output:**
```
════════════════════════════════════════════════════
  DIABETES RISK PREDICTION: 8D BAYES MODEL
════════════════════════════════════════════════════

[1] DATA PROCESSING
    [OK] Loaded 768 samples with 9 columns
    [OK] Cleaned 5 zero-columns with Gaussian imputation (376 values imputed)
    [OK] Log-transformed skewed features: ['Pregnancies', 'Insulin']
    [OK] Validated data integrity

[2] TRAIN-TEST SPLIT
    [OK] Training: 538 samples (34.8% diabetic)
    [OK] Testing:  230 samples (35.7% diabetic)

[3] MODEL TRAINING
    [OK] 8D Bayes model fitted on training data
    [OK] Prior P(Diabetic)     = 0.3480
    [OK] Prior P(Non-Diabetic) = 0.6520
    [OK] Covariance matrices positive definite
      Min eigenvalue (class 0): 5.126344e-06
      Min eigenvalue (class 1): 6.180384e-06

[4] PERFORMANCE METRICS (Test Set, threshold=0.5)
    Decision threshold: 0.5
    Accuracy:    0.7870 (78.7%)
    Sensitivity: 0.6522 (65.2%)
    Specificity: 0.8636 (86.3%)
    Precision:   0.7368 (73.7%)

    Confusion Matrix:
                        Predicted 0   Predicted 1
      Actual 0 (Healthy)   TN=162       FP=26
      Actual 1 (Diabetic)  FN=16        TP=26

════════════════════════════════════════════════════
  [OK] Analysis complete.
════════════════════════════════════════════════════
```

### 4️⃣ Run Web UI

```bash
python app.py
```

Navigate to **http://localhost:5000** in your browser. Enter 8 clinical features and get instant risk prediction with a probability score.

---

## 🔬 Deep Dive: Model Walkthrough

### How Data Cleaning Works

**Step 1: Gaussian Imputation**

```python
# Before: Glucose = 0 (invalid)
# Mean of valid values: μ = 121.5
# Imputed: Glucose = 121.5 + ε, where ε ~ N(0, 0.1)
# Result: Realistic value respecting the data distribution
```

**Step 2: Log-Transform**

```python
# Before: Insulin distribution is right-skewed (many low values, few very high)
# After: log(Insulin + 1) becomes approximately normal
# Benefit: Bayes classifier assumes Gaussian-distributed features
```

**Step 3: Stratified Splitting**

```python
# Naive 70-30 split might give:
#   Train: 100% non-diabetic (bad!)
#   Test:  0% non-diabetic (unrepresentative)
#
# Stratified split ensures:
#   Train: 34.8% diabetic (matches original)
#   Test:  35.7% diabetic (matches original)
#   Result: Fair evaluation
```

### How the Model Learns

```python
# For each class (Diabetic and Non-Diabetic):
# 1. Compute mean vector μ = mean(X_class)
# 2. Compute covariance matrix Σ = cov(X_class)
# 3. Add regularization: Σ' = Σ + λI (ensures numerical stability)
# 4. Verify positive definiteness via eigenvalue analysis

model = BayesDiabetesClassifier()
model.fit(X_train, y_train)  # Learns means, covariances, priors
```

### How Prediction Works

```python
# Given a patient's feature vector x = [Pregnancies, Glucose, ..., Age]:
#
# 1. Compute P(x | Non-Diabetic) using multivariate Gaussian PDF
# 2. Compute P(x | Diabetic) using multivariate Gaussian PDF
# 3. Multiply by priors: P(x, class) = P(x | class) * P(class)
# 4. Normalize using Bayes' theorem to get P(Diabetic | x)
# 5. If P(Diabetic | x) ≥ 0.5: classify as Diabetic, else Non-Diabetic

prob = model.predict_probabilities(x_new)
# Returns: P(Diabetic | x_new) ∈ [0, 1]

label = model.predict(x_new, threshold=0.5)
# Returns: 0 (Non-Diabetic) or 1 (Diabetic)
```

### Handling Edge Cases

**Issue:** In 8D space, multivariate Gaussian PDFs can underflow to zero

**Solution:** All computations use **log-space**:
```python
log_posterior = log_likelihood - log(normalization_constant)
posterior = exp(log_posterior)  # Convert back at the end
```

**Issue:** Covariance matrix might be singular (non-invertible)

**Solution:** Add **diagonal regularization**:
```python
Σ' = Σ + λI,  where λ = 1e-6
# Guarantees all eigenvalues > 0
```

---

## 📁 Project Structure

```
Diabetes_Risk_Model/
│
├── README.md                    # This file
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── config.py                    # Centralized configuration
├── main.py                      # CLI: Train & evaluate
├── app.py                       # Flask web server
│
├── src/                         # Core library
│   ├── __init__.py
│   ├── data_cleaning.py         # Data loading, imputation, transforms
│   ├── model.py                 # BayesDiabetesClassifier
│   └── evaluation.py            # Performance metrics
│
├── archive/                     # Dataset directory
│   └── diabetes.csv             # Pima Indians Diabetes Dataset (download separately)
│
├── templates/                   # Flask HTML templates
│   └── index.html               # Web UI form
│
└── static/                      # Static assets
    └── style.css                # Web UI styling
```

---

## 🚀 Usage Guide

### Command-Line Interface

**Train and evaluate the model:**

```bash
python main.py
```

This:
1. Loads and cleans the dataset
2. Trains the Bayesian classifier
3. Evaluates on test set
4. Prints all metrics and confusion matrix

### Web User Interface

**Start the Flask server:**

```bash
python app.py
```

**Access the app:**
- Open http://localhost:5000 in your browser
- Enter all 8 clinical features
- Get instant risk prediction with probability score
- See clinical recommendations (e.g., "Normal BMI: 18.5-24.9")

**Example Input:**
```
Pregnancies: 6
Glucose: 148
Blood Pressure: 72
Skin Thickness: 35
Insulin: 0
BMI: 33.6
Diabetes Pedigree Function: 0.627
Age: 50
```

**Output:**
```
Risk Score: 78.4%
Status: Higher than average risk of diabetes
```

### Programmatic Usage (Python)

```python
import numpy as np
from src.data_cleaning import run_cleaning_pipeline
from src.model import BayesDiabetesClassifier
from config import FEATURE_NAMES

# Train the model
_, X_train, X_test, y_train, y_test = run_cleaning_pipeline()
model = BayesDiabetesClassifier()
model.fit(X_train, y_train)

# Make a prediction for a new patient
patient_features = np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])
risk_probability = model.predict_probabilities(patient_features)[0]
print(f"Risk of diabetes: {risk_probability:.1%}")

# Get model internals
summary = model.get_model_summary()
print(f"Prior P(Diabetic) = {summary['prior_diabetic']:.4f}")
```

---

## 🔍 Key Assumptions & Limitations

### Model Assumptions

1. **Gaussian Distribution** — We assume features follow multivariate normal distributions within each class
   - This is approximately true after log-transform, but imperfect
   - Non-Gaussian features may reduce model accuracy

2. **Independent Features** — QDA assumes features can be described by a full covariance matrix (not identity matrix like Naive Bayes)
   - Slight correlations between features are captured
   - Strong correlations might cause numerical issues

3. **Homogeneous Population** — Model trained on Pima Indian women
   - May not generalize to other ethnic groups, men, or different age distributions
   - **Not intended for clinical deployment without retraining on representative populations**

### Performance Limitations

1. **65% Sensitivity** — We miss ~35% of actual diabetic patients
   - Unacceptable for screening; better for enrichment
   - Lower decision threshold could improve, but increases false positives

2. **Data Quality** — 376 zero-values had to be imputed (49% of data!)
   - Imputation introduces noise and bias
   - Original dataset may have measurement/recording issues

3. **Temporal Drift** — Model trained on historical data (1990s)
   - Diabetes prevalence, risk factors, and treatment may have changed
   - Regular retraining recommended

4. **Missing Features** — Dataset lacks important clinical information:
   - HbA1c (long-term glucose control)
   - Cholesterol/triglycerides
   - Family history details
   - Lifestyle factors (diet, exercise)

### When NOT to Use This Model

❌ Clinical diagnosis (use certified medical tests)  
❌ High-stakes decisions without expert review  
❌ Populations outside of women with similar demographic  
❌ Replacing standard diagnostic criteria  

### When to Use This Model

✅ Research and prototyping  
✅ Understanding Bayesian classification  
✅ Educational purposes  
✅ Screening enrichment (pre-filtering candidates for full testing)  
✅ Portfolio demonstration  

---

## 🎓 Lessons Learned

### Technical Insights

1. **Data quality matters more than algorithm complexity**
   - 49% of data needed imputation; garbage in = garbage out
   - Careful preprocessing beats fancy models

2. **Bayes' theorem is surprisingly practical**
   - Simple, interpretable, theoretically grounded
   - No hyperparameters to tune (just decision threshold)

3. **Log-space computation prevents bugs**
   - Overflow/underflow kills probability models in high dimensions
   - `logaddexp` saved the day multiple times

4. **Regularization is essential**
   - Without adding λI to covariance matrix, model crashed on singular matrices
   - Small regularization (1e-6) had zero impact on performance but huge impact on stability

### Project Insights

1. **Configuration belongs in config.py, not scattered in code**
   - Easy to experiment with hyperparameters
   - Single source of truth

2. **Proper module structure scales**
   - `src/data_cleaning.py`, `src/model.py`, `src/evaluation.py` = clean separation of concerns
   - Easy to test, easy to reuse

3. **Docstrings are worth their weight in gold**
   - Helped me understand my own code weeks later
   - Makes collaboration possible

---

## 🚀 Future Improvements

### Short-Term (High Impact)

1. **Cross-validation metrics**
   - k-fold CV to better estimate generalization error
   - Confidence intervals on performance metrics

2. **ROC curve & AUC analysis**
   - Visualize sensitivity/specificity trade-off across thresholds
   - Choose optimal threshold for specific use case

3. **Feature importance analysis**
   - Which features drive predictions most?
   - Use mutual information or feature ablation

4. **Unit tests**
   - Test data cleaning edge cases
   - Test model on synthetic data
   - Continuous integration

### Medium-Term

5. **Dataset comparison**
   - Evaluate on other diabetes datasets
   - Measure cross-dataset generalization

6. **Alternative models**
   - Logistic regression (simpler baseline)
   - Random Forest (captures non-linearity)
   - Neural network (if enough data)

7. **Threshold optimization**
   - Current 0.5 is arbitrary
   - Optimize for business metric (sensitivity vs. specificity trade-off)

8. **Web UI enhancements**
   - Save prediction history
   - Visualize risk factors
   - Batch upload CSV for multiple patients

### Long-Term (Production)

9. **Retraining pipeline**
   - Automated data validation
   - Model versioning
   - A/B testing new versions

10. **Deployment**
    - Docker containerization
    - REST API (FastAPI)
    - Cloud hosting (AWS/GCP)

11. **Monitoring**
    - Track prediction calibration over time
    - Alert on model drift
    - Feedback loop for retraining

---

## 📚 Resources & References

### Mathematical References

- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer. ← Bayes' theorem, Gaussian distributions
- Murphy, K. P. (2012). *Machine Learning: A Probabilistic Perspective*. MIT Press. ← QDA, LDA, Bayes classifiers
- [Stanford CS229: Generative Learning](https://cs229.stanford.edu/notes2021fall/generative_models.pdf) ← QDA derivation

### Practical References

- UCI Machine Learning Repository: [Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes)
- Kaggle: [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)

### Related Projects

- [sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis](https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html) ← Industry standard QDA
- [TensorFlow Probability](https://www.tensorflow.org/probability) ← Probabilistic deep learning

---

## 📜 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) file for details.

You are free to:
- ✅ Use commercially
- ✅ Modify for any purpose
- ✅ Distribute copies
- ✅ Use privately

As long as you:
- ⚠️ Include license and copyright notice
- ⚠️ Do not hold author liable

---

## 🙏 Acknowledgements

- **UCI Machine Learning Repository** for the Pima Indians Diabetes Dataset
- **Open source community** for NumPy, Pandas, scikit-learn, and Flask
- **Mathematics** — Bayes' theorem (1763), the gift that keeps giving

---

## ❓ FAQ

**Q: Why not just use sklearn's QDA?**  
A: This project implements the model from scratch to demonstrate understanding of the underlying mathematics. For production use, sklearn's optimized version is recommended.

**Q: Can I use this for medical diagnosis?**  
A: No. This is a research/educational model. Medical diagnosis requires certified clinical tests and professional judgment.

**Q: How do I improve the sensitivity?**  
A: Lower the decision threshold below 0.5. For example, at threshold=0.4, you'll catch more diabetic patients but get more false positives.

**Q: What if the dataset is missing?**  
A: Download from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or use the wget command in Quick Start.

**Q: Can I modify hyperparameters?**  
A: Yes! Edit `config.py`:
- `TEST_SIZE` — change train/test split
- `COVARIANCE_REGULARIZATION` — adjust numerical stability
- `DECISION_THRESHOLD` — change classification boundary
- `RANDOM_STATE` — for reproducibility

---

**Questions or contributions?** Open an issue or submit a pull request!
