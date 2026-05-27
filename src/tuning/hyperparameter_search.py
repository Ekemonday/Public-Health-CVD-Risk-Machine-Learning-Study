"""
Hyperparameter Tuning Module
GridSearchCV & RandomizedSearchCV for all models
"""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold


def tune_regression(model, X_train, y_train, method='random', cv=5):
    """Tune a regression model. Returns best estimator."""

    param_grids = {
        'Linear Regression (Ridge)': {
            'alpha': [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]
        },
        'Lasso Regression': {
            'alpha': [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
        },
        'XGBoost Regressor': {
            'n_estimators':     [100, 200, 300],
            'max_depth':        [4, 5, 6, 7],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8],
            'min_child_weight': [1, 3, 5],
        }
    }

    grid = param_grids.get(model.name, {})
    if not grid:
        print(f"No tuning grid for {model.name}. Skipping.")
        return model.model

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    if method == 'random':
        search = RandomizedSearchCV(
            model.model, grid, n_iter=30, cv=kf,
            scoring='r2', n_jobs=-1, random_state=42, verbose=1
        )
    else:
        search = GridSearchCV(
            model.model, grid, cv=kf,
            scoring='r2', n_jobs=-1, verbose=1
        )

    search.fit(X_train, y_train)
    print(f"\n[{model.name}] Best params: {search.best_params_}")
    print(f"[{model.name}] Best CV R²: {search.best_score_:.4f}")
    model.model = search.best_estimator_
    model.is_fitted = True
    return model


def tune_classification(model, X_train, y_train, method='random', cv=5):
    """Tune a classification model. Returns best estimator."""

    param_grids = {
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1.0, 5.0, 10.0],
        },
        'XGBoost Classifier': {
            'n_estimators':     [100, 200, 300],
            'max_depth':        [3, 4, 5, 6],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8],
            'scale_pos_weight': [1.5, 2.0, 2.3, 3.0],
        }
    }

    grid = param_grids.get(model.name, {})
    if not grid:
        print(f"No tuning grid for {model.name}. Skipping.")
        return model

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    if method == 'random':
        search = RandomizedSearchCV(
            model.model, grid, n_iter=30, cv=skf,
            scoring='roc_auc', n_jobs=-1, random_state=42, verbose=1
        )
    else:
        search = GridSearchCV(
            model.model, grid, cv=skf,
            scoring='roc_auc', n_jobs=-1, verbose=1
        )

    search.fit(X_train, y_train)
    print(f"\n[{model.name}] Best params: {search.best_params_}")
    print(f"[{model.name}] Best CV AUC: {search.best_score_:.4f}")
    model.model = search.best_estimator_
    model.is_fitted = True
    return model
