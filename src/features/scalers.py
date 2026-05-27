"""
Feature Scaling Module
"""

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
import joblib
import os


def scale_features(X_train, X_val, X_test, save_path=None):
    # Impute remaining NaNs with median, then standard-scale
    imputer = SimpleImputer(strategy='median')
    scaler  = StandardScaler()

    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_val_imp   = pd.DataFrame(imputer.transform(X_val),       columns=X_val.columns,   index=X_val.index)
    X_test_imp  = pd.DataFrame(imputer.transform(X_test),      columns=X_test.columns,  index=X_test.index)

    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_imp), columns=X_train.columns, index=X_train.index)
    X_val_scaled   = pd.DataFrame(scaler.transform(X_val_imp),       columns=X_val.columns,   index=X_val.index)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test_imp),      columns=X_test.columns,  index=X_test.index)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(imputer, os.path.join(save_path, 'imputer.pkl'))
        joblib.dump(scaler,  os.path.join(save_path, 'scaler.pkl'))
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
