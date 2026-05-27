"""
Tests — Data pipeline, features, model I/O
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from src.data.data_preprocessing import load_data, handle_missing_values, encode_categoricals, add_engineered_features
from src.data.data_splitting import split_features_targets, split_data


DATA_PATH = "data/raw/public_health_dataset.csv"


class TestDataLoading:
    def test_load_data_shape(self):
        df = load_data(DATA_PATH)
        assert df.shape[0] >= 1000, "Dataset must have at least 1000 rows"
        assert df.shape[1] >= 30, "Dataset must have at least 30 columns"

    def test_required_columns(self):
        df = load_data(DATA_PATH)
        required = ['age', 'bmi', 'systolic_bp', 'blood_glucose_mgdl',
                    'cvd_risk_score', 'hospitalized_30days']
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_target_ranges(self):
        df = load_data(DATA_PATH)
        assert df['cvd_risk_score'].between(1, 40).all(), "CVD risk scores out of range"
        assert df['hospitalized_30days'].isin([0, 1]).all(), "Hospitalization must be binary"


class TestPreprocessing:
    def setup_method(self):
        self.df = load_data(DATA_PATH)

    def test_missing_value_imputation(self):
        df_clean = handle_missing_values(self.df)
        lab_cols = ['hba1c_percent', 'ldl_cholesterol_mgdl',
                    'triglycerides_mgdl', 'creatinine_mgdl']
        for col in lab_cols:
            assert df_clean[col].isnull().sum() == 0, f"Nulls remain in {col}"

    def test_encoding_no_nans(self):
        df_clean = handle_missing_values(self.df)
        df_enc = encode_categoricals(df_clean)
        assert df_enc.isnull().sum().sum() == 0, "Encoding produced NaN values"

    def test_feature_engineering(self):
        df_clean = handle_missing_values(self.df)
        df_enc = encode_categoricals(df_clean)
        df_feat = add_engineered_features(df_enc)
        assert 'pulse_pressure' in df_feat.columns
        assert 'chol_ratio' in df_feat.columns
        assert 'metabolic_risk' in df_feat.columns


class TestDataSplitting:
    def setup_method(self):
        from src.data.data_preprocessing import preprocess
        self.df = preprocess(DATA_PATH)

    def test_split_sizes(self):
        X, y_reg, y_cls = split_features_targets(self.df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_reg)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(X), "Split sizes don't add up"

    def test_no_target_leakage(self):
        X, y_reg, y_cls = split_features_targets(self.df)
        assert 'cvd_risk_score' not in X.columns
        assert 'hospitalized_30days' not in X.columns

    def test_no_index_overlap(self):
        X, y_reg, y_cls = split_features_targets(self.df)
        X_train, X_val, X_test, *_ = split_data(X, y_reg)
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0


class TestModels:
    def setup_method(self):
        from src.data.data_preprocessing import preprocess
        from src.data.data_splitting import split_features_targets, split_data
        from src.features.scalers import scale_features
        df = preprocess(DATA_PATH)
        X, y_reg, y_cls = split_features_targets(df)
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y_reg)
        X_train_s, X_val_s, X_test_s, _ = scale_features(X_train, X_val, X_test)
        self.X_train, self.y_train = X_train_s, y_train
        self.X_test, self.y_test = X_test_s, y_test
        self.y_cls_train = y_cls.iloc[y_train.index - y_train.index.min()] if hasattr(y_cls, 'iloc') else y_cls

    def test_ridge_regression_output(self):
        from src.models.regression_models import LinearRegressionModel
        m = LinearRegressionModel().build()
        m.fit(self.X_train, self.y_train)
        preds = m.predict(self.X_test)
        assert len(preds) == len(self.X_test)
        assert preds.dtype in [float, np.float32, np.float64]

    def test_xgboost_regressor_output(self):
        from src.models.xgboost_models import XGBoostRegressorModel
        m = XGBoostRegressorModel(n_estimators=50).build()
        m.fit(self.X_train, self.y_train)
        preds = m.predict(self.X_test)
        assert len(preds) == len(self.X_test)

    def test_model_save_load(self, tmp_path):
        from src.models.regression_models import LinearRegressionModel
        m = LinearRegressionModel().build()
        m.fit(self.X_train, self.y_train)
        path = m.save(str(tmp_path))
        m2 = LinearRegressionModel().build()
        m2.load(path)
        preds1 = m.predict(self.X_test)
        preds2 = m2.predict(self.X_test)
        np.testing.assert_array_almost_equal(preds1, preds2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
