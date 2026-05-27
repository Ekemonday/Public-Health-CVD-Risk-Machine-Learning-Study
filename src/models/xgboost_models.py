"""
XGBoost Models Module
- XGBoost Regressor  (CVD Risk Score)
- XGBoost Classifier (30-day Hospitalization)
"""

from xgboost import XGBRegressor, XGBClassifier
from .base_model import BaseModel


class XGBoostRegressorModel(BaseModel):
    """
    XGBoost Gradient Boosting Regressor for CVD Risk Score.
    Handles non-linear relationships and interactions automatically.
    """

    def __init__(self, **params):
        super().__init__("XGBoost Regressor")
        self.params = {
            'n_estimators':     300,
            'max_depth':        6,
            'learning_rate':    0.05,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma':            0.1,
            'reg_alpha':        0.1,
            'reg_lambda':       1.0,
            'objective':        'reg:squarederror',
            'eval_metric':      'rmse',
            'random_state':     42,
            'n_jobs':           -1,
            'verbosity':        0,
        }
        self.params.update(params)

    def build(self, **kwargs):
        self.params.update(kwargs)
        self.model = XGBRegressor(**self.params)
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"\n[{self.name}] Training on {X_train.shape[0]} samples...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        print(f"[{self.name}] Training complete.")
        return self


class XGBoostClassifierModel(BaseModel):
    """
    XGBoost Gradient Boosting Classifier for 30-day Hospitalization.
    Uses scale_pos_weight to handle class imbalance.
    """

    def __init__(self, scale_pos_weight=2.3, **params):
        super().__init__("XGBoost Classifier")
        self.params = {
            'n_estimators':      300,
            'max_depth':         5,
            'learning_rate':     0.05,
            'subsample':         0.8,
            'colsample_bytree':  0.8,
            'min_child_weight':  3,
            'gamma':             0.1,
            'reg_alpha':         0.1,
            'reg_lambda':        1.0,
            'scale_pos_weight':  scale_pos_weight,
            'objective':         'binary:logistic',
            'eval_metric':       'auc',
            'use_label_encoder': False,
            'random_state':      42,
            'n_jobs':            -1,
            'verbosity':         0,
        }
        self.params.update(params)

    def build(self, **kwargs):
        self.params.update(kwargs)
        # Remove unsupported param in newer XGBoost
        self.params.pop('use_label_encoder', None)
        self.model = XGBClassifier(**self.params)
        return self

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        print(f"\n[{self.name}] Training on {X_train.shape[0]} samples...")
        eval_set = [(X_val, y_val)] if X_val is not None else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False
        )
        self.is_fitted = True
        print(f"[{self.name}] Training complete.")
        return self
