"""
Data Splitting Module
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_features_targets(df: pd.DataFrame):
    regression_target = 'cvd_risk_score'
    classification_target = 'hospitalized_30days'
    drop_cols = [regression_target, classification_target]
    X = df.drop(columns=drop_cols, errors='ignore')
    y_reg = df[regression_target]
    y_cls = df[classification_target]
    return X, y_reg, y_cls


def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42, stratify=None):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    val_ratio = val_size / (1 - test_size)
    strat_temp = stratify.iloc[X_temp.index] if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_state,
        stratify=strat_temp
    )
    print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    return X_train, X_val, X_test, y_train, y_val, y_test
