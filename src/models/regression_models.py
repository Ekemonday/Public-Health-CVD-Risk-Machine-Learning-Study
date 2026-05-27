"""
Regression Models Module
- Linear Regression (CVD Risk Score prediction)
- Logistic Regression (Hospitalization classification)
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .base_model import BaseModel


class LinearRegressionModel(BaseModel):
    """
    Linear Regression for CVD Risk Score prediction.
    Uses Ridge regularization to reduce multicollinearity.
    """

    def __init__(self, alpha=1.0):
        super().__init__("Linear Regression (Ridge)")
        self.alpha = alpha

    def build(self, **kwargs):
        self.alpha = kwargs.get('alpha', self.alpha)
        self.model = Ridge(alpha=self.alpha, random_state=42)
        return self


class LassoRegressionModel(BaseModel):
    """
    Lasso Regression — performs implicit feature selection.
    """

    def __init__(self, alpha=0.1):
        super().__init__("Lasso Regression")
        self.alpha = alpha

    def build(self, **kwargs):
        self.alpha = kwargs.get('alpha', self.alpha)
        self.model = Lasso(alpha=self.alpha, max_iter=5000, random_state=42)
        return self


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression for 30-day Hospitalization prediction.
    """

    def __init__(self, C=1.0, max_iter=1000):
        super().__init__("Logistic Regression")
        self.C = C
        self.max_iter = max_iter

    def build(self, **kwargs):
        self.C = kwargs.get('C', self.C)
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42
        )
        return self
