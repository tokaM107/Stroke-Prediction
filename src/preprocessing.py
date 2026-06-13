import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StrokeFeatureEngineer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.bmi_median_ = X["bmi"].median()

        return self

    def transform(self, X):

        X = X.copy()

        X.drop(columns=["id"], errors="ignore", inplace=True)

        X["bmi"] = X["bmi"].fillna(self.bmi_median_)

        X["avg_glucose_level"] = np.log1p(
            X["avg_glucose_level"]
        )

        X["health_factor"] = (
            X["hypertension"]
            + X["heart_disease"]
            + (
                X["avg_glucose_level"]
                > np.log1p(140)
            ).astype(int)
        )

        return X