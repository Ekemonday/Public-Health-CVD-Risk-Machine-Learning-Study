"""
============================================================
  PUBLIC HEALTH CVD RISK ML STUDY — MAIN PIPELINE
  Study: Cardiovascular Disease Risk & Hospitalization
  Models: Linear/Lasso Regression + XGBoost (both tasks)
============================================================
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ── imports ──────────────────────────────────────────────
from src.data.data_preprocessing import preprocess
from src.data.data_splitting import split_features_targets, split_data
from src.features.scalers import scale_features
from src.models.regression_models import LinearRegressionModel, LassoRegressionModel, LogisticRegressionModel
from src.models.xgboost_models import XGBoostRegressorModel, XGBoostClassifierModel
from src.training.trainer import train_regression_model, train_classification_model, cross_validate_model
from src.utils.visualization import (
    plot_target_distributions, plot_correlation_heatmap, plot_model_comparison
)
from src.utils.logger import get_logger

logger = get_logger()


def main():
    logger.info("=" * 60)
    logger.info("PUBLIC HEALTH ML STUDY — PIPELINE START")
    logger.info("=" * 60)

    # ── 1. Load & preprocess ──────────────────────────────
    logger.info("Step 1: Loading and preprocessing data...")
    df_raw = pd.read_csv("data/raw/public_health_dataset.csv")
    df = preprocess("data/raw/public_health_dataset.csv")
    logger.info(f"Processed dataset shape: {df.shape}")

    # ── 2. EDA plots ──────────────────────────────────────
    logger.info("Step 2: Generating EDA visualizations...")
    plot_target_distributions(df_raw['cvd_risk_score'], df_raw['hospitalized_30days'])
    plot_correlation_heatmap(df_raw)

    # ── 3. Split features / targets ───────────────────────
    logger.info("Step 3: Splitting data...")
    X, y_reg, y_cls = split_features_targets(df)

    X_train_r, X_val_r, X_test_r, y_train_r, y_val_r, y_test_r = split_data(
        X, y_reg, test_size=0.20, val_size=0.10)
    X_train_c, X_val_c, X_test_c, y_train_c, y_val_c, y_test_c = split_data(
        X, y_cls, test_size=0.20, val_size=0.10, stratify=y_cls)

    # ── 4. Scale features ─────────────────────────────────
    logger.info("Step 4: Scaling features...")
    Xtr_r, Xvr, Xte_r, scaler = scale_features(X_train_r, X_val_r, X_test_r, save_path="models_saved")
    Xtr_c, Xvc, Xte_c, _      = scale_features(X_train_c, X_val_c, X_test_c)

    # ──────────────────────────────────────────────────────
    # TASK A — REGRESSION: CVD Risk Score
    # ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("TASK A: Regression — CVD Risk Score Prediction")
    logger.info("=" * 50)
    regression_results = {}

    # --- Model A1: Ridge Regression ---
    logger.info("\n[A1] Ridge Regression")
    ridge = LinearRegressionModel(alpha=1.0).build()
    cross_validate_model(ridge, Xtr_r, y_train_r, task='regression')
    m_ridge = regression_results['Ridge Regression'] = train_regression_model(
        ridge, Xtr_r, y_train_r, Xvr, y_val_r, Xte_r, y_test_r)

    # --- Model A2: Lasso Regression ---
    logger.info("\n[A2] Lasso Regression")
    lasso = LassoRegressionModel(alpha=0.1).build()
    cross_validate_model(lasso, Xtr_r, y_train_r, task='regression')
    m_lasso = regression_results['Lasso Regression'] = train_regression_model(
        lasso, Xtr_r, y_train_r, Xvr, y_val_r, Xte_r, y_test_r)

    # --- Model A3: XGBoost Regressor ---
    logger.info("\n[A3] XGBoost Regressor")
    xgb_reg = XGBoostRegressorModel().build()
    m_xgb_r = regression_results['XGBoost Regressor'] = train_regression_model(
        xgb_reg, Xtr_r, y_train_r, Xvr, y_val_r, Xte_r, y_test_r)

    # ──────────────────────────────────────────────────────
    # TASK B — CLASSIFICATION: 30-Day Hospitalization
    # ──────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("TASK B: Classification — 30-Day Hospitalization")
    logger.info("=" * 50)
    classification_results = {}

    # --- Model B1: Logistic Regression ---
    logger.info("\n[B1] Logistic Regression")
    log_reg = LogisticRegressionModel(C=1.0).build()
    cross_validate_model(log_reg, Xtr_c, y_train_c, task='classification')
    m_log = classification_results['Logistic Regression'] = train_classification_model(
        log_reg, Xtr_c, y_train_c, Xvc, y_val_c, Xte_c, y_test_c)

    # --- Model B2: XGBoost Classifier ---
    logger.info("\n[B2] XGBoost Classifier")
    xgb_cls = XGBoostClassifierModel().build()
    m_xgb_c = classification_results['XGBoost Classifier'] = train_classification_model(
        xgb_cls, Xtr_c, y_train_c, Xvc, y_val_c, Xte_c, y_test_c)

    # ── 5. Compare all models ─────────────────────────────
    logger.info("\nStep 5: Model comparison plots...")
    all_results = {**regression_results, **classification_results}
    plot_model_comparison(all_results)

    # ── 6. Save summary JSON ──────────────────────────────
    logger.info("Step 6: Saving results summary...")
    summary = {
        "study": "CVD Risk & Hospitalization Prediction",
        "dataset_rows": len(df),
        "features": X.shape[1],
        "regression_results": {
            k: {m: v for m, v in metrics.items()}
            for k, metrics in regression_results.items()
        },
        "classification_results": {
            k: {m: str(v) for m, v in metrics.items()}
            for k, metrics in classification_results.items()
        }
    }
    os.makedirs("reports", exist_ok=True)
    with open("reports/results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # ── 7. Print final leaderboard ────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("FINAL MODEL LEADERBOARD")
    logger.info("=" * 60)
    logger.info("\n REGRESSION — CVD Risk Score")
    logger.info(f"  {'Model':<30} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    logger.info(f"  {'-'*56}")
    for name, m in regression_results.items():
        logger.info(f"  {name:<30} {m['R2']:>8.4f} {m['RMSE']:>8.4f} {m['MAE']:>8.4f}")

    logger.info("\n CLASSIFICATION — 30-Day Hospitalization")
    logger.info(f"  {'Model':<30} {'AUC':>8} {'F1':>8} {'Acc':>8}")
    logger.info(f"  {'-'*56}")
    for name, m in classification_results.items():
        logger.info(f"  {name:<30} {str(m['AUC']):>8} {m['F1']:>8.4f} {m['Accuracy']:>8.4f}")

    logger.info("\n Pipeline complete. Check reports/ for plots & models_saved/ for artifacts.")
    return summary


if __name__ == "__main__":
    main()
