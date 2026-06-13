import streamlit as st
import pandas as pd
import joblib
import sys

sys.path.append("src")
from src.preprocessing import StrokeFeatureEngineer

st.set_page_config(
    page_title="Stroke Risk Prediction",
    page_icon="🩺",
    layout="centered"
)

pipeline = joblib.load("models/stroke_pipeline.pkl")

st.title("🩺 Stroke Risk Prediction")
st.write(
    "Enter patient information and estimate the probability of stroke."
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

age = st.slider(
    "Age",
    min_value=18,
    max_value=100,
    value=50
)

hypertension = st.selectbox(
    "Hypertension",
    [0, 1]
)

heart_disease = st.selectbox(
    "Heart Disease",
    [0, 1]
)

ever_married = st.selectbox(
    "Ever Married",
    ["Yes", "No"]
)

work_type = st.selectbox(
    "Work Type",
    [
        "Private",
        "Self-employed",
        "Govt_job",
        "children",
        "Never_worked"
    ]
)

residence = st.selectbox(
    "Residence Type",
    ["Urban", "Rural"]
)

avg_glucose_level = st.number_input(
    "Average Glucose Level",
    min_value=50.0,
    max_value=400.0,
    value=100.0
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=80.0,
    value=25.0
)

smoking_status = st.selectbox(
    "Smoking Status",
    [
        "never smoked",
        "formerly smoked",
        "smokes",
        "Unknown"
    ]
)

if st.button("Predict"):

    patient = pd.DataFrame(
        [{
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence,
            "avg_glucose_level": avg_glucose_level,
            "bmi": bmi,
            "smoking_status": smoking_status
        }]
    )

    prediction = pipeline.predict(patient)[0]

    probability = pipeline.predict_proba(patient)[0][1]

    st.subheader("Result")

    st.metric(
        "Stroke Probability",
        f"{probability:.2%}"
    )

    if prediction == 1:
        st.error("⚠️ High Stroke Risk")
    else:
        st.success("✅ Low Stroke Risk")