"""
End-to-End Campus Energy Intelligence Pipeline
Orchestrates data acquisition, feature engineering, model training,
comparative evaluation, anomaly detection, explainability, and artifact persistence.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from src.data_generator import generate_campus_dataset, BUILDING_PROFILES
from src.preprocessing import engineer_features, get_feature_columns, train_val_test_split
from src.models import train_and_evaluate_all, calculate_metrics
from src.anomaly_detector import EnergyAnomalyDetector, evaluate_anomaly_detection
from src.explainability import EnergyExplainer


class CampusEnergyPipeline:
    """
    Complete end-to-end training and inference manager.
    """
    def __init__(self, data_dir: str = "data", model_dir: str = "saved_models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        self.raw_data: pd.DataFrame = None
        self.featured_data: pd.DataFrame = None
        self.train_df: pd.DataFrame = None
        self.val_df: pd.DataFrame = None
        self.test_df: pd.DataFrame = None
        self.feature_cols: list = []
        self.target_col: str = "total_power_kwh"
        
        self.models: Dict[str, Any] = {}
        self.comparison_df: pd.DataFrame = None
        self.best_model_name: str = ""
        self.best_model: Any = None
        
        self.anomaly_detector: EnergyAnomalyDetector = None
        self.anomaly_metrics: Dict[str, float] = {}
        self.explainer: EnergyExplainer = None

    def run_full_pipeline(self, num_days: int = 730, seed: int = 42) -> Dict[str, Any]:
        """
        Executes the entire ML pipeline from data generation to artifact saving.
        """
        print("\n=======================================================")
        print("  STEP 1: GENERATING CAMPUS ENERGY & SENSOR DATASET   ")
        print("=======================================================")
        self.raw_data = generate_campus_dataset(
            start_date="2024-01-01 00:00:00",
            end_date="2025-12-31 23:00:00",
            seed=seed,
            anomaly_rate=0.03
        )
        raw_path = os.path.join(self.data_dir, "raw_campus_energy_data.csv")
        self.raw_data.to_csv(raw_path, index=False)
        print(f"[OK] Generated {len(self.raw_data):,} records across {self.raw_data['building_id'].nunique()} buildings.")
        print(f"[OK] Saved raw data to {raw_path}")

        print("\n=======================================================")
        print("  STEP 2: FEATURE ENGINEERING & PREPROCESSING        ")
        print("=======================================================")
        self.featured_data = engineer_features(self.raw_data)
        self.feature_cols, self.target_col = get_feature_columns()
        processed_path = os.path.join(self.data_dir, "featured_campus_energy_data.csv")
        self.featured_data.to_csv(processed_path, index=False)
        print(f"[OK] Engineered {len(self.feature_cols)} predictive features (lags, rolling stats, cyclic, thermal).")
        print(f"[OK] Saved featured data to {processed_path}")

        print("\n=======================================================")
        print("  STEP 3: CHRONOLOGICAL TRAIN / VAL / TEST SPLIT      ")
        print("=======================================================")
        self.train_df, self.val_df, self.test_df = train_val_test_split(self.featured_data)
        print(f"[OK] Train set: {len(self.train_df):,} samples ({len(self.train_df)/len(self.featured_data)*100:.1f}%)")
        print(f"[OK] Validation set: {len(self.val_df):,} samples ({len(self.val_df)/len(self.featured_data)*100:.1f}%)")
        print(f"[OK] Test set: {len(self.test_df):,} samples ({len(self.test_df)/len(self.featured_data)*100:.1f}%)")

        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df[self.target_col]
        X_val = self.val_df[self.feature_cols]
        y_val = self.val_df[self.target_col]
        X_test = self.test_df[self.feature_cols]
        y_test = self.test_df[self.target_col]

        print("\n=======================================================")
        print("  STEP 4: MODEL DEVELOPMENT & COMPARATIVE BENCHMARK   ")
        print("=======================================================")
        self.models, self.comparison_df, test_preds = train_and_evaluate_all(
            X_train, y_train, X_val, y_val, X_test, y_test
        )
        print("\nModel Benchmark Leaderboard:")
        print(self.comparison_df.to_string(index=False))
        
        # Select best performing model
        self.best_model_name = self.comparison_df.iloc[0]["Model"]
        self.best_model = self.models[self.best_model_name]
        best_preds = test_preds[self.best_model_name]
        print(f"\n[BEST MODEL] -> {self.best_model_name} (Test RMSE: {self.comparison_df.iloc[0]['Test RMSE (kWh)']} kWh, R2: {self.comparison_df.iloc[0]['Test R2']:.4f})")

        print("\n=======================================================")
        print("  STEP 5: ANOMALY DETECTION & RISK SCORING ENGINE     ")
        print("=======================================================")
        self.anomaly_detector = EnergyAnomalyDetector(contamination=0.03)
        self.anomaly_detector.fit(self.train_df)
        
        # Detect anomalies on test set using hybrid forecast residual + Isolation Forest
        test_with_anomalies = self.anomaly_detector.detect_anomalies(self.test_df, best_preds)
        self.anomaly_metrics = evaluate_anomaly_detection(test_with_anomalies)
        print(f"[OK] Anomaly Detection Precision: {self.anomaly_metrics['Precision']:.4f}, Recall: {self.anomaly_metrics['Recall']:.4f}, F1: {self.anomaly_metrics['F1_Score']:.4f}")
        
        # Save test results with anomaly flags
        test_res_path = os.path.join(self.data_dir, "test_predictions_with_anomalies.csv")
        test_with_anomalies.to_csv(test_res_path, index=False)

        print("\n=======================================================")
        print("  STEP 6: EXPLAINABLE AI & SHAP VALUES COMPUTATION    ")
        print("=======================================================")
        # Fit SHAP explainer on XGBoost or LightGBM model for tree speed
        tree_model_name = "XGBoost Regressor" if "XGBoost Regressor" in self.models else self.best_model_name
        tree_model = self.models.get(tree_model_name, self.best_model)
        self.explainer = EnergyExplainer(tree_model, self.feature_cols)
        shap_vals, shap_importance = self.explainer.explain_global(X_test, max_samples=400)
        shap_imp_path = os.path.join(self.data_dir, "shap_feature_importance.csv")
        shap_importance.to_csv(shap_imp_path, index=False)
        print(f"[OK] Top 5 global energy drivers according to SHAP:\n{shap_importance.head(5).to_string(index=False)}")

        print("\n=======================================================")
        print("  STEP 7: SERIALIZING ALL ARTIFACTS & METADATA        ")
        print("=======================================================")
        # Save models dict
        models_path = os.path.join(self.model_dir, "campus_energy_models.joblib")
        joblib.dump({
            "models": self.models,
            "best_model_name": self.best_model_name,
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "building_profiles": BUILDING_PROFILES
        }, models_path)
        
        # Save anomaly detector
        anom_path = os.path.join(self.model_dir, "energy_anomaly_detector.joblib")
        joblib.dump(self.anomaly_detector, anom_path)
        
        # Save metrics summary
        metrics_summary = {
            "best_model": self.best_model_name,
            "model_comparison": self.comparison_df.to_dict(orient="records"),
            "anomaly_metrics": self.anomaly_metrics,
            "num_records_total": len(self.featured_data),
            "num_test_records": len(self.test_df),
            "feature_count": len(self.feature_cols)
        }
        with open(os.path.join(self.data_dir, "metrics_summary.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)
            
        print(f"[OK] All models saved to {models_path}")
        print(f"[OK] Anomaly detector saved to {anom_path}")
        print(f"[OK] Pipeline execution successfully completed!")
        
        return metrics_summary


def load_pipeline_artifacts(data_dir: str = "data", model_dir: str = "saved_models") -> Dict[str, Any]:
    """Loads all trained pipeline artifacts for Streamlit app consumption."""
    models_path = os.path.join(model_dir, "campus_energy_models.joblib")
    anom_path = os.path.join(model_dir, "energy_anomaly_detector.joblib")
    
    saved_model_dict = joblib.load(models_path)
    anomaly_detector = joblib.load(anom_path)
    
    featured_df = pd.read_csv(os.path.join(data_dir, "featured_campus_energy_data.csv"))
    featured_df["timestamp"] = pd.to_datetime(featured_df["timestamp"])
    
    test_results_df = pd.read_csv(os.path.join(data_dir, "test_predictions_with_anomalies.csv"))
    test_results_df["timestamp"] = pd.to_datetime(test_results_df["timestamp"])
    
    shap_importance_df = pd.read_csv(os.path.join(data_dir, "shap_feature_importance.csv"))
    
    with open(os.path.join(data_dir, "metrics_summary.json"), "r") as f:
        metrics_summary = json.load(f)
        
    return {
        "models": saved_model_dict["models"],
        "best_model_name": saved_model_dict["best_model_name"],
        "feature_cols": saved_model_dict["feature_cols"],
        "target_col": saved_model_dict["target_col"],
        "building_profiles": saved_model_dict["building_profiles"],
        "anomaly_detector": anomaly_detector,
        "featured_df": featured_df,
        "test_results_df": test_results_df,
        "shap_importance_df": shap_importance_df,
        "metrics_summary": metrics_summary
    }
