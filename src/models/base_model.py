"""
Base Model — Abstract class all models inherit from
Public Health CVD Risk Study
"""

from abc import ABC, abstractmethod
import joblib
import os


class BaseModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_fitted = False

    @abstractmethod
    def build(self, **kwargs):
        """Instantiate the underlying estimator."""
        pass

    def fit(self, X_train, y_train):
        print(f"\n[{self.name}] Training on {X_train.shape[0]} samples...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        print(f"[{self.name}] Training complete.")
        return self

    def predict(self, X):
        assert self.is_fitted, f"{self.name} is not fitted yet."
        return self.model.predict(X)

    def predict_proba(self, X):
        assert self.is_fitted
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        raise NotImplementedError(f"{self.name} has no predict_proba.")

    def save(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{self.name.lower().replace(' ', '_')}.pkl")
        joblib.dump(self.model, path)
        print(f"[{self.name}] Saved to {path}")
        return path

    def load(self, path: str):
        self.model = joblib.load(path)
        self.is_fitted = True
        print(f"[{self.name}] Loaded from {path}")
        return self

    def get_feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            return dict(zip(feature_names, self.model.feature_importances_))
        if hasattr(self.model, 'coef_'):
            import numpy as np
            coefs = self.model.coef_.ravel() if self.model.coef_.ndim > 1 else self.model.coef_
            return dict(zip(feature_names, abs(coefs)))
        return {}
