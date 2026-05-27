"""
Metrics & Evaluation Module
"""

import numpy as np
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)


def regression_metrics(y_true, y_pred, label="") -> dict:
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))) * 100
    metrics = {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "MAPE": round(mape, 4)}
    print(f"\n{'='*40}")
    print(f"REGRESSION METRICS {label}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    return metrics


def classification_metrics(y_true, y_pred, y_prob=None, label="") -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    metrics = {
        "Accuracy": round(acc, 4), "Precision": round(prec, 4),
        "Recall": round(rec, 4), "F1": round(f1, 4),
        "AUC": round(auc, 4) if auc else "N/A"
    }
    print(f"\n{'='*40}")
    print(f"CLASSIFICATION METRICS {label}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Hospitalized','Hospitalized']))
    return metrics
