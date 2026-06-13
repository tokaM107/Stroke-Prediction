import joblib
import pandas as pd

pipeline = joblib.load(
    "models/stroke_pipeline.pkl"
)

patient = pd.DataFrame(
    [
        {
            "gender": "Male",
            "age": 67,
            "hypertension": 1,
            "heart_disease": 0,
            "ever_married": "Yes",
            "work_type": "Private",
            "Residence_type": "Urban",
            "avg_glucose_level": 180,
            "bmi": 31,
            "smoking_status": "formerly smoked",
        }
    ]
)

prediction = pipeline.predict(patient)

probability = pipeline.predict_proba(
    patient
)[:, 1]

print(prediction)
print(probability)