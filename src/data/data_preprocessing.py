"""
Data Preprocessing Module
Public Health CVD Risk Study
"""

import pandas as pd
import numpy as np


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def describe_data(df: pd.DataFrame):
    print("\n=== Dataset Overview ===")
    print(df.describe(include='all').T[['count', 'unique', 'top', 'mean', 'std', 'min', 'max']])
    print("\n=== Missing Values ===")
    miss = df.isnull().sum()
    print(miss[miss > 0])


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lab_cols = ['hba1c_percent', 'ldl_cholesterol_mgdl', 'triglycerides_mgdl', 'creatinine_mgdl']
    for col in lab_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    cat_cols = ['education_level', 'alcohol_consumption', 'insurance_coverage',
                'smoking_status', 'physical_activity_level', 'diet_quality',
                'medication_adherence', 'gender', 'urban_rural', 'region', 'marital_status']
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ordinal_maps = {
        'education_level':        {'None': 0, 'Primary': 1, 'Secondary': 2, 'Tertiary': 3},
        'income_level':           {'Low': 0, 'Middle': 1, 'High': 2},
        'physical_activity_level':{'Sedentary': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
        'diet_quality':           {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
        'medication_adherence':   {'Poor': 0, 'Moderate': 1, 'Good': 2},
        'insurance_coverage':     {'None': 0, 'Partial': 1, 'Full': 2},
        'smoking_status':         {'Never': 0, 'Former': 1, 'Current': 2},
        'alcohol_consumption':    {'None': 0, 'Moderate': 1, 'Heavy': 2},
    }
    for col, mapping in ordinal_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    binary_maps = {
        'gender':     {'Male': 1, 'Female': 0},
        'urban_rural': {'Urban': 2, 'Suburban': 1, 'Rural': 0},
    }
    for col, mapping in binary_maps.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    nominal_cols = ['region', 'marital_status']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pulse_pressure'] = df['systolic_bp'] - df['diastolic_bp']
    df['chol_ratio'] = df['total_cholesterol_mgdl'] / (df['hdl_cholesterol_mgdl'] + 1)
    df['glucose_bmi'] = df['blood_glucose_mgdl'] * df['bmi']
    df['age_bmi'] = df['age'] * df['bmi']
    df['metabolic_risk'] = df['hypertension'] + df['diabetes'] + (df['bmi'] > 30).astype(int)
    return df


def preprocess(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    df = add_engineered_features(df)
    df.drop(columns=['patient_id'], inplace=True, errors='ignore')
    return df
