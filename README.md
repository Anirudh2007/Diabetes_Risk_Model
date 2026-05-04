# Diabetes_Risk_Model

DSC 10 Diabetes Risk Prediction Model — end semester project.

## Overview

An 8-dimensional Bayesian classifier (Quadratic Discriminant Analysis) for predicting diabetes risk from the UCI Pima Indians Diabetes Dataset. The model applies Bayes' theorem with full-covariance multivariate Gaussian class-conditionals.

## Pipeline (`python main.py`)

1. Data loading and Gaussian zero-imputation of biologically impossible values
2. Log-transform of right-skewed features (Pregnancies, Insulin)
3. Stratified 70/30 train-test split
4. 8D Bayes classifier training
5. Evaluation at decision threshold = 0.5: accuracy, sensitivity, specificity, precision, and confusion matrix

## Web UI (`python app.py`)

Flask app on port 5000. Enter all 8 clinical features; the model outputs the posterior probability P(Diabetic | x) and a risk message (≥ 50% = higher risk).
