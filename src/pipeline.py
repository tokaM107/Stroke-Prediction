from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from preprocessing import StrokeFeatureEngineer


def build_final_pipeline():

    numeric_features = [
        "age",
        "bmi",
        "avg_glucose_level",
    ]

    categorical_features = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                ),
                categorical_features,
            ),
        ],
        remainder="passthrough",
    )

    model = VotingClassifier(
        estimators=[
            (
                "xgb",
                XGBClassifier(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    scale_pos_weight=10,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
    )

    pipeline = Pipeline(
        [
            (
                "feature_engineering",
                StrokeFeatureEngineer(),
            ),
            (
                "preprocessing",
                preprocessor,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    return pipeline