"""
Inference Module
Load saved models and generate predictions on new data
"""

import pandas as pd
import numpy as np
import joblib
import os


def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def load_scaler(scaler_path: str = "models_saved/scaler.pkl"):
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    return joblib.load(scaler_path)


def predict_cvd_risk(model, scaler, patient_data: pd.DataFrame) -> np.ndarray:
    """
    Predict 10-year CVD Risk Score for new patients.
    patient_data: DataFrame with same feature columns as training set.
    """
    X_scaled = scaler.transform(patient_data)
    predictions = model.predict(X_scaled)
    return np.clip(predictions, 1, 40)


def predict_hospitalization(model, scaler, patient_data: pd.DataFrame):
    """
    Predict 30-day hospitalization probability and binary label.
    Returns (labels, probabilities)
    """
    X_scaled = scaler.transform(patient_data)
    probs = model.predict_proba(X_scaled)[:, 1]
    labels = (probs >= 0.5).astype(int)
    return labels, probs


def predict_risk_category(cvd_score: float) -> str:
    """Convert numeric CVD score to clinical risk category."""
    if cvd_score < 7.5:
        return "Low Risk"
    elif cvd_score < 15:
        return "Borderline Risk"
    elif cvd_score < 25:
        return "Intermediate Risk"
    else:
        return "High Risk"


def batch_predict(reg_model_path: str, cls_model_path: str, scaler_path: str,
                  input_csv: str, output_csv: str):
    """
    End-to-end batch prediction on a CSV file.
    Saves results to output_csv.
    """
    df = pd.read_csv(input_csv)
    scaler = load_scaler(scaler_path)
    reg_model = load_model(reg_model_path)
    cls_model = load_model(cls_model_path)

    X = df.copy()
    if 'patient_id' in X.columns:
        ids = X.pop('patient_id')
    else:
        ids = pd.Series(range(len(X)), name='patient_id')

    for col in ['cvd_risk_score', 'hospitalized_30days']:
        X.drop(columns=[col], inplace=True, errors='ignore')

    X_scaled = scaler.transform(X)
    cvd_scores = np.clip(reg_model.predict(X_scaled), 1, 40)
    hosp_labels, hosp_probs = (
        (cls_model.predict(X_scaled),
         cls_model.predict_proba(X_scaled)[:, 1])
        if hasattr(cls_model, 'predict_proba')
        else (cls_model.predict(X_scaled), None)
    )

    results = pd.DataFrame({
        'patient_id': ids.values,
        'predicted_cvd_risk_score': cvd_scores.round(2),
        'cvd_risk_category': [predict_risk_category(s) for s in cvd_scores],
        'hospitalization_predicted': hosp_labels,
        'hospitalization_probability': hosp_probs.round(4) if hosp_probs is not None else None,
    })

    results.to_csv(output_csv, index=False)
    print(f"Batch predictions saved to {output_csv}")
    print(results.head(10).to_string())
    return results
