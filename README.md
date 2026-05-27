#  Public Health CVD Risk — Machine Learning Study

**Study:** Cardiovascular Disease Risk Score Prediction & 30-Day Hospitalization Classification  
**Dataset:** 2,000 patient records · 39 features · two prediction targets  
**Models:** Ridge Regression · Lasso Regression · Logistic Regression · XGBoost (×2)

---

##  Project Structure

```
ml_public_health/
├── data/
│   └── raw/
│       ├── public_health_dataset.csv     ← 2,000-row dataset (39 cols)
│       └── generate_dataset.py           ← dataset generator script
├── src/
│   ├── data/
│   │   ├── data_preprocessing.py         ← imputation, encoding, feature engineering
│   │   └── data_splitting.py             ← train/val/test splits
│   ├── features/
│   │   └── scalers.py                    ← StandardScaler + SimpleImputer pipeline
│   ├── models/
│   │   ├── base_model.py                 ← abstract base class
│   │   ├── regression_models.py          ← Ridge, Lasso, Logistic Regression
│   │   └── xgboost_models.py             ← XGBoost Regressor & Classifier
│   ├── training/
│   │   └── trainer.py                    ← fit, evaluate, cross-validate
│   ├── evaluation/
│   │   └── metrics.py                    ← MAE/RMSE/R² · AUC/F1/Accuracy
│   ├── tuning/
│   │   └── hyperparameter_search.py      ← RandomizedSearchCV / GridSearchCV
│   ├── inference/
│   │   └── predict.py                    ← batch prediction on new data
│   └── utils/
│       ├── visualization.py              ← all plots (EDA, ROC, residuals, etc.)
│       ├── logger.py                     ← file + console logging
│       └── config.py                     ← JSON config loader
├── configs/
│   └── config.json                       ← all hyperparameters & paths
├── models_saved/                         ← trained model .pkl files + scaler
├── reports/                              ← all PNG charts + results JSON
├── tests/
│   └── test_pipeline.py                  ← 12 unit tests (all pass )
└── main.py                               ← run full pipeline end-to-end
```

---

##  Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn joblib

# Run the full pipeline
python main.py
```

---

##  Dataset Features (39 columns)

| Category | Features |
|----------|---------|
| Demographics | age, gender, region, urban_rural, education_level, income_level, marital_status |
| Physical | bmi, systolic_bp, diastolic_bp, heart_rate, waist_circumference_cm |
| Lab Results | blood_glucose, hba1c, total_cholesterol, ldl, hdl, triglycerides, creatinine |
| Lifestyle | smoking_status, alcohol_consumption, physical_activity_level, diet_quality, sleep_hours, stress_level |
| Medical History | hypertension, diabetes, family_history_cvd, previous_heart_attack, stroke_history, kidney_disease |
| Healthcare | insurance_coverage, distance_to_clinic_km, hospital_visits_per_year, medication_adherence |
| **Targets** | **cvd_risk_score** (regression 1–40), **hospitalized_30days** (binary classification) |

---

## 🤖 Models & Results

### Task A — CVD Risk Score Regression

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Ridge Regression | 0.8779 | 2.1307 | 1.7157 |
| Lasso Regression | **0.8825** | **2.0903** | **1.6804** |
| XGBoost Regressor | 0.8477 | 2.3800 | 1.9249 |

### Task B — 30-Day Hospitalization Classification

| Model | AUC | F1 | Accuracy |
|-------|-----|----|---------|
| Logistic Regression | 0.7978 | 0.6159 | 0.7225 |
| XGBoost Classifier | **0.8158** | 0.5984 | **0.7450** |

---

## 📈 Generated Report Charts

- `target_distributions.png` — CVD risk histogram + hospitalization bar chart
- `correlation_heatmap.png` — feature correlation matrix
- `regression_diagnostics_*.png` — predicted vs actual + residual plots
- `roc_curve_*.png` — ROC curves with AUC
- `confusion_matrix_*.png` — classification confusion matrices
- `feature_importance_*.png` — top 20 most important features per model
- `model_comparison.png` — side-by-side R² and AUC bar chart
- `results_summary.json` — all metrics in JSON format

---

## 🧪 Tests

```bash
pip install pytest
python -m pytest tests/test_pipeline.py -v
# 12 passed 
```

Tests cover: data loading · missing value imputation · encoding · feature engineering ·
train/val/test splits · target leakage check · model output shapes · model save/load.

---

##  Configuration

Edit `configs/config.json` to change hyperparameters, paths, or toggle tuning.

---

##  Inference on New Data

```python
from src.inference.predict import batch_predict

batch_predict(
    reg_model_path  = "models_saved/lasso_regression.pkl",
    cls_model_path  = "models_saved/xgboost_classifier.pkl",
    scaler_path     = "models_saved/scaler.pkl",
    input_csv       = "new_patients.csv",
    output_csv      = "reports/predictions.csv"
)
```
# Public-Health-CVD-Risk-Machine-Learning-Study
