"""
Visualization Module — EDA, training curves, feature importance
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


PALETTE = "Blues_d"
FIG_DIR = "reports"
os.makedirs(FIG_DIR, exist_ok=True)


def plot_target_distributions(y_reg, y_cls, save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Public Health Study — Target Variable Distributions", fontsize=15, fontweight='bold')

    axes[0].hist(y_reg, bins=40, color='steelblue', edgecolor='white', alpha=0.85)
    axes[0].set_title("CVD Risk Score (Regression Target)")
    axes[0].set_xlabel("10-Year CVD Risk Score")
    axes[0].set_ylabel("Count")
    axes[0].axvline(y_reg.mean(), color='red', linestyle='--', label=f"Mean={y_reg.mean():.1f}")
    axes[0].legend()

    cls_counts = y_cls.value_counts()
    bars = axes[1].bar(['Not Hospitalized', 'Hospitalized'], cls_counts.values,
                       color=['#4a90d9', '#e05c5c'], edgecolor='white')
    axes[1].set_title("30-Day Hospitalization (Classification Target)")
    axes[1].set_ylabel("Count")
    for bar, val in zip(bars, cls_counts.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                     f'{val}\n({val/len(y_cls)*100:.1f}%)', ha='center', fontsize=11)

    plt.tight_layout()
    if save:
        path = f"{FIG_DIR}/target_distributions.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def plot_correlation_heatmap(df, save=True):
    num_df = df.select_dtypes(include=[np.number]).drop(
        columns=['cvd_risk_score', 'hospitalized_30days'], errors='ignore')
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(18, 14))
    sns.heatmap(corr, cmap='RdBu_r', center=0, linewidths=0.3, annot=False,
                square=True, ax=ax, fmt='.2f', vmin=-1, vmax=1)
    ax.set_title("Feature Correlation Matrix — Public Health Dataset", fontsize=14, fontweight='bold', pad=12)
    plt.tight_layout()
    if save:
        path = f"{FIG_DIR}/correlation_heatmap.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def plot_feature_importance(importances: dict, model_name: str, top_n=20, save=True):
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    features, values = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(features)))[::-1]
    bars = ax.barh(features, values, color=colors, edgecolor='white')
    ax.invert_yaxis()
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Importance Score")
    for bar, val in zip(bars, values):
        ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(' ', '_')
        path = f"{FIG_DIR}/feature_importance_{fname}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def plot_regression_results(y_true, y_pred, model_name: str, save=True):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Regression Diagnostics — {model_name}", fontsize=13, fontweight='bold')

    axes[0].scatter(y_pred, y_true, alpha=0.4, color='steelblue', s=20)
    mn, mx = min(y_pred.min(), y_true.min()), max(y_pred.max(), y_true.max())
    axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect fit')
    axes[0].set_xlabel("Predicted CVD Risk")
    axes[0].set_ylabel("Actual CVD Risk")
    axes[0].set_title("Predicted vs Actual")
    axes[0].legend()

    axes[1].scatter(y_pred, residuals, alpha=0.4, color='darkorange', s=20)
    axes[1].axhline(0, color='red', linestyle='--', lw=2)
    axes[1].set_xlabel("Predicted CVD Risk")
    axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")

    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(' ', '_')
        path = f"{FIG_DIR}/regression_diagnostics_{fname}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def plot_roc_curve(y_true, y_prob, model_name: str, save=True):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='steelblue', lw=2.5, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
    ax.fill_between(fpr, tpr, alpha=0.1, color='steelblue')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}", fontsize=13, fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(' ', '_')
        path = f"{FIG_DIR}/roc_curve_{fname}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name: str, save=True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Hosp.', 'Hospitalized'],
                yticklabels=['Not Hosp.', 'Hospitalized'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight='bold')
    plt.tight_layout()
    if save:
        fname = model_name.lower().replace(' ', '_')
        path = f"{FIG_DIR}/confusion_matrix_{fname}.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()


def plot_model_comparison(results: dict, save=True):
    """Bar chart comparing all models."""
    reg_models = {k: v for k, v in results.items() if 'R2' in v}
    cls_models = {k: v for k, v in results.items() if 'AUC' in v}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Model Comparison — Public Health ML Study", fontsize=14, fontweight='bold')

    if reg_models:
        names = list(reg_models.keys())
        r2s = [reg_models[n]['R2'] for n in names]
        bars = axes[0].bar(names, r2s, color=['#4a90d9', '#e07b54'], edgecolor='white', width=0.5)
        axes[0].set_title("Regression: R² Score")
        axes[0].set_ylim(0, 1.05)
        axes[0].set_ylabel("R²")
        for bar, val in zip(bars, r2s):
            axes[0].text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.4f}', ha='center', fontsize=11)

    if cls_models:
        names = list(cls_models.keys())
        aucs = [cls_models[n]['AUC'] for n in names]
        bars = axes[1].bar(names, aucs, color=['#5cb85c', '#d9534f'], edgecolor='white', width=0.5)
        axes[1].set_title("Classification: AUC Score")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_ylabel("AUC-ROC")
        for bar, val in zip(bars, aucs):
            axes[1].text(bar.get_x()+bar.get_width()/2, val+0.01, f'{val:.4f}', ha='center', fontsize=11)

    plt.tight_layout()
    if save:
        path = f"{FIG_DIR}/model_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    plt.close()
