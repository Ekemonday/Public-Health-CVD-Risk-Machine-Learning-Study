"""
Training Pipeline Module
Handles training, evaluation, and saving for all models
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

from src.evaluation.metrics import regression_metrics, classification_metrics
from src.utils.visualization import (
    plot_regression_results, plot_roc_curve,
    plot_confusion_matrix, plot_feature_importance
)


def train_regression_model(model, X_train, y_train, X_val, y_val,
                            X_test, y_test, save_dir="models_saved"):
    """
    Full training + evaluation cycle for a regression model.
    Returns dict of test metrics.
    """
    # Train
    if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    # Val metrics
    val_pred = model.predict(X_val)
    print(f"\n--- Validation ---")
    regression_metrics(y_val, val_pred, label=f"[{model.name}] Val")

    # Test metrics
    test_pred = model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred, label=f"[{model.name}] Test")

    # Plots
    plot_regression_results(y_test, pd.Series(test_pred, index=y_test.index), model.name)

    # Feature importance
    fi = model.get_feature_importance(X_train.columns.tolist())
    if fi:
        plot_feature_importance(fi, model.name, top_n=20)

    # Save model
    model.save(save_dir)

    return test_metrics


def train_classification_model(model, X_train, y_train, X_val, y_val,
                                X_test, y_test, save_dir="models_saved"):
    """
    Full training + evaluation cycle for a classification model.
    Returns dict of test metrics.
    """
    # Train
    if hasattr(model, 'fit') and 'X_val' in model.fit.__code__.co_varnames:
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        model.fit(X_train, y_train)

    # Val metrics
    val_pred = model.predict(X_val)
    val_prob = model.predict_proba(X_val)
    print(f"\n--- Validation ---")
    classification_metrics(y_val, val_pred, val_prob, label=f"[{model.name}] Val")

    # Test metrics
    test_pred = model.predict(X_test)
    test_prob = model.predict_proba(X_test)
    test_metrics = classification_metrics(y_test, test_pred, test_prob,
                                          label=f"[{model.name}] Test")

    # Plots
    plot_roc_curve(y_test, test_prob, model.name)
    plot_confusion_matrix(y_test, test_pred, model.name)

    # Feature importance
    fi = model.get_feature_importance(X_train.columns.tolist())
    if fi:
        plot_feature_importance(fi, model.name, top_n=20)

    # Save model
    model.save(save_dir)

    return test_metrics


def cross_validate_model(model, X, y, task='regression', cv=5):
    """
    Run cross-validation and print fold-level scores.
    """
    scoring = 'r2' if task == 'regression' else 'roc_auc'
    kf = KFold(n_splits=cv, shuffle=True, random_state=42) if task == 'regression' \
        else StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    scores = cross_val_score(model.model, X, y, cv=kf, scoring=scoring, n_jobs=-1)
    print(f"\n[{model.name}] {cv}-Fold CV ({scoring}): "
          f"{scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Folds: {[round(s,4) for s in scores]}")
    return scores
