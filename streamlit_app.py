import streamlit as st
import numpy as np
import pandas as pd

from config import FEATURE_NAMES, LOG_TRANSFORM_OFFSET, ZERO_INVALID_COLUMNS
from src.data_cleaning import run_cleaning_pipeline
from src.model import BayesDiabetesClassifier

# Set Streamlit page config
st.set_page_config(
    page_title="Diabetes Risk Checker",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching modern design system
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #102a43;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #334e68;
        margin-bottom: 2rem;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2563eb 0%, #0f172a 100%);
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 14px;
        padding: 0.75rem 1rem;
        border: none;
        box-shadow: 0 10px 20px rgba(37, 99, 235, 0.2);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #1e293b 100%);
        color: white;
    }
    .result-card-high {
        background-color: rgba(254, 226, 226, 0.95);
        border: 2px solid #ef4444;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 12px 30px rgba(239, 68, 68, 0.15);
    }
    .result-card-low {
        background-color: rgba(209, 250, 229, 0.95);
        border: 2px solid #10b981;
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.15);
    }
    .risk-score {
        font-size: 3.5rem;
        font-weight: 900;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Train Bayes classifier model once using cache
@st.cache_resource
def load_and_train_model():
    _, X_train, X_test, y_train, y_test = run_cleaning_pipeline()
    model = BayesDiabetesClassifier()
    model.fit(X_train, y_train)

    training_df = pd.DataFrame(X_train, columns=FEATURE_NAMES)
    training_means = training_df.mean().to_dict()
    return model, training_means

model, training_means = load_and_train_model()

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies (count)",
    "Glucose": "Glucose (mg/dL)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin (mu U/ml)",
    "BMI": "Body Mass Index (BMI)",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
}

FEATURE_OPTIMALS = {
    "Pregnancies": "Optimal: 0-2 (lower is better)",
    "Glucose": "Optimal: <100 mg/dL (normal fasting)",
    "BloodPressure": "Optimal: <120 mm Hg (normal)",
    "SkinThickness": "Optimal: 10-20 mm (normal range)",
    "Insulin": "Optimal: 5-15 mu U/ml (normal fasting)",
    "BMI": "Optimal: 18.5-24.9 (normal)",
    "DiabetesPedigreeFunction": "Optimal: <0.5 (lower family history)",
    "Age": "Optimal: Younger is lower risk",
}

# Header UI
st.markdown('<div class="main-title">🩺 Diabetes Risk Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Enter medical details to estimate diabetes risk probability using the 8D Bayesian Classifier.</div>', unsafe_allow_html=True)

# Form layout
with st.form("risk_form"):
    st.subheader("Medical Details")
    col1, col2 = st.columns(2)

    inputs = {}

    with col1:
        inputs["Pregnancies"] = st.number_input(
            FEATURE_LABELS["Pregnancies"],
            min_value=0, max_value=20, value=1, step=1,
            help=FEATURE_OPTIMALS["Pregnancies"]
        )
        inputs["Glucose"] = st.number_input(
            FEATURE_LABELS["Glucose"],
            min_value=0.0, max_value=300.0, value=110.0, step=1.0,
            help=FEATURE_OPTIMALS["Glucose"]
        )
        inputs["BloodPressure"] = st.number_input(
            FEATURE_LABELS["BloodPressure"],
            min_value=0.0, max_value=200.0, value=70.0, step=1.0,
            help=FEATURE_OPTIMALS["BloodPressure"]
        )
        inputs["SkinThickness"] = st.number_input(
            FEATURE_LABELS["SkinThickness"],
            min_value=0.0, max_value=100.0, value=20.0, step=1.0,
            help=FEATURE_OPTIMALS["SkinThickness"]
        )

    with col2:
        inputs["Insulin"] = st.number_input(
            FEATURE_LABELS["Insulin"],
            min_value=0.0, max_value=900.0, value=79.0, step=1.0,
            help=FEATURE_OPTIMALS["Insulin"]
        )
        inputs["BMI"] = st.number_input(
            FEATURE_LABELS["BMI"],
            min_value=0.0, max_value=80.0, value=25.0, step=0.1,
            help=FEATURE_OPTIMALS["BMI"]
        )
        inputs["DiabetesPedigreeFunction"] = st.number_input(
            FEATURE_LABELS["DiabetesPedigreeFunction"],
            min_value=0.0, max_value=3.0, value=0.47, step=0.01,
            help=FEATURE_OPTIMALS["DiabetesPedigreeFunction"]
        )
        inputs["Age"] = st.number_input(
            FEATURE_LABELS["Age"],
            min_value=1, max_value=120, value=30, step=1,
            help=FEATURE_OPTIMALS["Age"]
        )

    submitted = st.form_submit_button("Compute Diabetes Risk")

if submitted:
    # Process inputs
    values = []
    for feature in FEATURE_NAMES:
        val = float(inputs[feature])
        if feature in ZERO_INVALID_COLUMNS and val == 0.0:
            val = float(training_means.get(feature, 1.0))
        values.append(val)

    df_user = pd.DataFrame([values], columns=FEATURE_NAMES)
    for feature in ["Pregnancies", "Insulin"]:
        df_user[feature] = np.log(df_user[feature] + LOG_TRANSFORM_OFFSET)

    prob = float(model.predict_probabilities(df_user.values)[0]) * 100.0

    st.markdown("---")
    if prob >= 50.0:
        st.markdown(
            f"""
            <div class="result-card-high">
                <h3 style="color: #991b1b; margin: 0;">High Risk Detected</h3>
                <div class="risk-score" style="color: #dc2626;">{prob:.1f}%</div>
                <p style="color: #7f1d1d; font-weight: 600; font-size: 1.1rem; margin: 0;">Higher than average risk of diabetes.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="result-card-low">
                <h3 style="color: #065f46; margin: 0;">Low Risk Detected</h3>
                <div class="risk-score" style="color: #059669;">{prob:.1f}%</div>
                <p style="color: #047857; font-weight: 600; font-size: 1.1rem; margin: 0;">Lower than average risk of diabetes.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

st.markdown("<br><hr><div style='text-align: center; color: #64748b; font-size: 0.85rem;'>Powered by the 8D Bayesian Diabetes Risk Model</div>", unsafe_allow_html=True)
