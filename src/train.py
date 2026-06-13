import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from pipeline import build_final_pipeline

RANDOM_STATE = 42

df = pd.read_csv("data/raw.csv")

X = df.drop(columns=["stroke"])
y = df["stroke"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,
)

pipeline = build_final_pipeline()

pipeline.fit(X_train, y_train)

joblib.dump(
    pipeline,
    "models/stroke_pipeline.pkl",
)

print("Pipeline saved successfully.")