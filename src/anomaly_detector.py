"""
Anomaly Detection & Energy Risk Scoring Engine
Combines Isolation Forest, statistical forecasting residual analysis,
automated root-cause diagnostics, and energy wastage cost calculation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class EnergyAnomalyDetector:
    """
    Hybrid Anomaly Detector combining Isolation Forest and
    Supervised Forecast Residuals to detect campus energy wastage.
    """
    def __init__(self, contamination: float = 0.03, tariff_per_kwh: float = 8.5, co2_kg_per_kwh: float = 0.82):
        self.contamination = contamination
        self.tariff_per_kwh = tariff_per_kwh
        self.co2_kg_per_kwh = co2_kg_per_kwh
        self.iso_forest = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self.feature_cols = [
            "total_power_kwh", "temperature_c", "occupancy_rate",
            "hour", "is_weekend", "is_business_hour", "cooling_degree_hours"
        ]

    def fit(self, df: pd.DataFrame):
        """Fits the Isolation Forest on standard operational records."""
        X = df[self.feature_cols].copy()
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        return self

    def detect_anomalies(
        self,
        df: pd.DataFrame,
        y_pred: np.ndarray = None
    ) -> pd.DataFrame:
        """
        Detects anomalies using Isolation Forest and forecasting residuals.
        Assigns severity, root cause explanation, and wastage cost estimation.
        """
        results = df.copy()
        X = results[self.feature_cols].copy()
        X_scaled = self.scaler.transform(X)
        
        # 1. Isolation Forest scores
        # decision_function returns negative values for anomalies
        iso_scores = self.iso_forest.decision_function(X_scaled)
        iso_preds = self.iso_forest.predict(X_scaled)  # -1 for anomaly, 1 for normal
        
        results["iso_anomaly_score"] = np.round(-iso_scores, 3)  # higher is more anomalous
        results["is_iso_anomaly"] = (iso_preds == -1).astype(int)
        
        # 2. Residual Analysis
        if y_pred is not None:
            results["forecast_kwh"] = np.round(y_pred, 2)
            residuals = results["total_power_kwh"] - y_pred
            results["residual_kwh"] = np.round(residuals, 2)
            
            res_std = np.std(residuals) if np.std(residuals) > 0 else 1.0
            results["residual_zscore"] = np.round(residuals / res_std, 2)
        else:
            results["forecast_kwh"] = results["total_power_kwh"]
            results["residual_kwh"] = 0.0
            results["residual_zscore"] = 0.0

        # 3. Hybrid Confidence & Severity Assignment
        severities = []
        is_detected_list = []
        root_causes = []
        recommendations = []
        wastage_kwh_list = []
        wastage_cost_list = []
        co2_excess_list = []
        
        for idx, row in results.iterrows():
            power = row["total_power_kwh"]
            pred = row.get("forecast_kwh", power)
            z = row.get("residual_zscore", 0.0)
            iso_flag = row["is_iso_anomaly"]
            hour = int(row["hour"])
            occ = row["occupancy_rate"]
            is_wk = int(row["is_weekend"])
            temp = row["temperature_c"]
            b_type = row.get("building_type", "Academic")
            
            # Anomaly trigger criteria
            is_anom = False
            severity = "Normal"
            cause = "Normal Operations"
            rec = "No action required."
            
            excess_kwh = max(0.0, power - pred)
            
            if (abs(z) >= 3.5) or (iso_flag == 1 and abs(z) >= 2.0):
                is_anom = True
                if abs(z) >= 4.0 or (power > pred * 1.8 and occ < 0.1):
                    severity = "Critical"
                elif abs(z) >= 2.8:
                    severity = "High"
                else:
                    severity = "Medium"
            elif (iso_flag == 1 and abs(z) >= 1.5) or (abs(z) >= 2.5):
                is_anom = True
                severity = "Low"
                
            # Root Cause Diagnostic Logic
            if is_anom:
                if (hour < 6 or hour > 21 or is_wk == 1) and occ < 0.15 and power > (pred + 15):
                    cause = "Off-Hours Energy Leakage (HVAC/Lighting active in empty building)"
                    rec = "Inspect automated building management timer relays and shut off idle zone HVAC."
                elif z > 3.0 and occ < 0.3:
                    cause = "Sudden Heavy Equipment Overload / Unscheduled Activity"
                    rec = "Check high-load equipment circuits and verify sub-meter power draw."
                elif z < -2.5 and occ > 0.4:
                    cause = "Abnormal Power Drop / Substation Sensor Malfunction"
                    rec = "Verify Phase line voltage and CT sensor telemetry calibration."
                elif temp > 35.0 and z > 2.0:
                    cause = "Thermal Strain / HVAC Efficiency Degradation Under Extreme Heat"
                    rec = "Service chiller condenser coils and adjust thermostat deadbands."
                else:
                    cause = "Unusual Consumption Surge vs Historical Baseline"
                    rec = "Review building occupancy log and sub-meter diagnostics."
            
            wastage_cost = excess_kwh * self.tariff_per_kwh
            excess_co2 = excess_kwh * self.co2_kg_per_kwh
            
            severities.append(severity)
            is_detected_list.append(1 if is_anom else 0)
            root_causes.append(cause)
            recommendations.append(rec)
            wastage_kwh_list.append(round(excess_kwh, 2))
            wastage_cost_list.append(round(wastage_cost, 2))
            co2_excess_list.append(round(excess_co2, 2))
            
        results["detected_anomaly"] = is_detected_list
        results["detected_severity"] = severities
        results["root_cause"] = root_causes
        results["recommended_action"] = recommendations
        results["wasted_energy_kwh"] = wastage_kwh_list
        results["financial_loss_inr"] = wastage_cost_list
        results["excess_co2_kg"] = co2_excess_list
        
        return results


def evaluate_anomaly_detection(df_with_detected: pd.DataFrame) -> Dict[str, float]:
    """Computes precision, recall, and F1-score against ground truth anomalies."""
    y_true = df_with_detected["is_anomaly"].values
    y_pred = df_with_detected["detected_anomaly"].values
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)
    
    return {
        "Accuracy": round(float(accuracy), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1_Score": round(float(f1), 4),
        "True_Positives": int(tp),
        "False_Positives": int(fp),
        "False_Negatives": int(fn)
    }
