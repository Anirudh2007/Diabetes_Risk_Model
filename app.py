from flask import Flask, render_template, request
import numpy as np
import pandas as pd

from config import FEATURE_NAMES, LOG_TRANSFORM_OFFSET, ZERO_INVALID_COLUMNS
from src.data_cleaning import run_cleaning_pipeline
from src.model import BayesDiabetesClassifier

app = Flask(__name__)

# Train the Bayes classifier once when the app starts
_, X_train, X_test, y_train, y_test = run_cleaning_pipeline()
model = BayesDiabetesClassifier()
model.fit(X_train, y_train)

# Use means from the cleaned training data to impute any invalid zeros in user input
training_dataframe = pd.DataFrame(X_train, columns=FEATURE_NAMES)
training_means = training_dataframe.mean().to_dict()

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies (count)",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (mu U/ml)",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
}

FEATURE_OPTIMALS = {
    "Pregnancies": "0-2 (lower is better)",
    "Glucose": "<100 (normal fasting)",
    "BloodPressure": "<120 (normal)",
    "SkinThickness": "10-20 (normal range)",
    "Insulin": "5-15 (normal fasting)",
    "BMI": "18.5-24.9 (normal)",
    "DiabetesPedigreeFunction": "<0.5 (lower family history)",
    "Age": "Younger is better",
}


def preprocess_input(form_data: dict) -> np.ndarray:
    values = []

    for feature in FEATURE_NAMES:
        raw_value = form_data.get(feature)
        if raw_value is None or raw_value.strip() == "":
            raise ValueError(f"Missing value for {feature}.")

        try:
            value = float(raw_value)
        except ValueError:
            raise ValueError(f"The value for {feature} must be a number.")

        if feature in ZERO_INVALID_COLUMNS and value == 0.0:
            value = float(training_means.get(feature, 1.0))

        values.append(value)

    dataframe = pd.DataFrame([values], columns=FEATURE_NAMES)

    for feature in ["Pregnancies", "Insulin"]:
        dataframe[feature] = np.log(dataframe[feature] + LOG_TRANSFORM_OFFSET)

    return dataframe.values


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        try:
            user_input = preprocess_input(request.form)

            probability = float(model.predict_probabilities(user_input)[0]) * 100.0

            result = {
                "probability": probability,
                "message": (
                    "Higher than average risk of diabetes." if probability >= 50.0 else "Lower than average risk of diabetes."
                ),
            }
        except ValueError as exc:
            error = str(exc)
        except Exception:
            error = "Unable to compute risk. Please check your inputs and try again."

    return render_template(
        "index.html",
        feature_names=FEATURE_NAMES,
        feature_labels=FEATURE_LABELS,
        feature_optimals=FEATURE_OPTIMALS,
        result=result,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
