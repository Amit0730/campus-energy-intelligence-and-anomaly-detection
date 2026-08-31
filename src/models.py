"""
Supervised Learning Model Implementations & Benchmarking Suite
Includes Baseline, Ridge, Random Forest, XGBoost, LightGBM, and Stacking Ensemble.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb


class PersistenceBaseline(BaseEstimator, RegressorMixin):
    """
    Persistence / Lag-24 Baseline:
    Predicts power consumption from the same hour yesterday (lag 24h)
    or falls back to rolling mean.
    """
    def __init__(self, lag_feature_name: str = "power_lag_24h"):
        self.lag_feature_name = lag_feature_name
        self.feature_names_in_ = None
        self.lag_idx_ = None
        self.fallback_mean_ = 0.0

    def fit(self, X, y):
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            if self.lag_feature_name in self.feature_names_in_:
                self.lag_idx_ = self.feature_names_in_.index(self.lag_feature_name)
        self.fallback_mean_ = np.mean(y)
        return self

    def predict(self, X):
        if hasattr(X, "values"):
            X_mat = X.values
        else:
            X_mat = np.asarray(X)
            
        if self.lag_idx_ is not None and self.lag_idx_ < X_mat.shape[1]:
            preds = X_mat[:, self.lag_idx_].copy()
            nan_mask = np.isnan(preds)
            preds[nan_mask] = self.fallback_mean_
            return preds
        return np.full(len(X_mat), self.fallback_mean_)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates comprehensive regression evaluation metrics."""
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)
    
    # Avoid zero division in MAPE
    mask = y_t > 0.01
    y_t_safe = y_t[mask]
    y_p_safe = y_p[mask]
    
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    mae = mean_absolute_error(y_t, y_p)
    r2 = r2_score(y_t, y_p)
    mape = np.mean(np.abs((y_t_safe - y_p_safe) / y_t_safe)) * 100.0
    
    # Directional Accuracy (hourly sign change agreement)
    if len(y_t) > 1:
        diff_true = np.diff(y_t)
        diff_pred = np.diff(y_p)
        dir_acc = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100.0
    else:
        dir_acc = 100.0
        
    return {
        "RMSE": round(float(rmse), 3),
        "MAE": round(float(mae), 3),
        "MAPE_pct": round(float(mape), 2),
        "R2_Score": round(float(r2), 4),
        "Directional_Accuracy_pct": round(float(dir_acc), 2)
    }


def get_model_zoo() -> Dict[str, Any]:
    """Returns a dictionary of all models to benchmark."""
    models = {
        "Persistence Baseline": PersistenceBaseline(),
        
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0, random_state=42))
        ]),
        
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=16,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        ),
        
        "LightGBM Regressor": lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1
        ),
        
        "XGBoost Regressor": xgb.XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            tree_method="hist",
            verbosity=0
        )
    }
    
    # Stacking Ensemble combining Ridge, Random Forest, LightGBM, and XGBoost
    estimators = [
        ("ridge", Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0, random_state=42))])),
        ("rf", RandomForestRegressor(n_estimators=60, max_depth=12, random_state=42, n_jobs=-1)),
        ("lgb", lgb.LGBMRegressor(n_estimators=120, learning_rate=0.05, num_leaves=31, random_state=42, verbosity=-1)),
        ("xgb", xgb.XGBRegressor(n_estimators=120, learning_rate=0.05, max_depth=5, random_state=42, tree_method="hist", verbosity=0))
    ]
    
    models["Stacking Ensemble"] = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1
    )
    
    return models


def train_and_evaluate_all(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Trains all models in the zoo, records training times,
    evaluates on validation and test sets, and compiles comparison table.
    """
    zoo = get_model_zoo()
    trained_models = {}
    test_predictions = {}
    comparison_records = []
    
    for name, model in zoo.items():
        print(f"--> Training {name}...")
        start_t = time.time()
        model.fit(X_train, y_train)
        fit_time = round(time.time() - start_t, 2)
        
        trained_models[name] = model
        
        # Predictions
        pred_train = model.predict(X_train)
        pred_val = model.predict(X_val)
        pred_test = model.predict(X_test)
        test_predictions[name] = pred_test
        
        # Metrics
        train_m = calculate_metrics(y_train, pred_train)
        val_m = calculate_metrics(y_val, pred_val)
        test_m = calculate_metrics(y_test, pred_test)
        
        comparison_records.append({
            "Model": name,
            "Train R2": train_m["R2_Score"],
            "Val R2": val_m["R2_Score"],
            "Test R2": test_m["R2_Score"],
            "Test RMSE (kWh)": test_m["RMSE"],
            "Test MAE (kWh)": test_m["MAE"],
            "Test MAPE (%)": test_m["MAPE_pct"],
            "Directional Acc (%)": test_m["Directional_Accuracy_pct"],
            "Train Time (s)": fit_time
        })
        
    comp_df = pd.DataFrame(comparison_records).sort_values("Test RMSE (kWh)").reset_index(drop=True)
    return trained_models, comp_df, test_predictions
