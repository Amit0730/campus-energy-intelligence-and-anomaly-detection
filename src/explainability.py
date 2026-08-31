"""
Explainable AI (XAI) Engine using SHAP
Provides global feature importance rankings and local instance-level
waterfall/force explanations for campus energy consumption predictions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import shap


class EnergyExplainer:
    """
    SHAP-based Model Explainability Engine for Tree Models (XGBoost / LightGBM / RF).
    """
    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        
        # Handle Pipeline unwrapping if needed
        unwrapped_model = model
        if hasattr(model, "named_steps"):
            unwrapped_model = model.named_steps["regressor"]
            
        try:
            self.explainer = shap.TreeExplainer(unwrapped_model)
        except Exception:
            self.explainer = shap.Explainer(unwrapped_model)

    def explain_global(self, X_sample: pd.DataFrame, max_samples: int = 500) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Computes SHAP values across a representative sample and returns
        global feature importance DataFrame.
        """
        if len(X_sample) > max_samples:
            sample_df = X_sample.sample(max_samples, random_state=42)
        else:
            sample_df = X_sample
            
        shap_values = self.explainer.shap_values(sample_df)
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        
        importance_df = pd.DataFrame({
            "Feature": self.feature_names,
            "Mean_Abs_SHAP": np.round(mean_abs_shap, 4)
        }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
        
        return shap_values, importance_df

    def explain_instance(self, single_row: pd.Series) -> Dict[str, Any]:
        """
        Explains a single hourly forecast instance.
        Returns base value, predicted impact, and top driving factors.
        """
        row_df = pd.DataFrame([single_row[self.feature_names]])
        shap_vals = self.explainer.shap_values(row_df)[0]
        
        base_val = getattr(self.explainer, "expected_value", 0.0)
        if isinstance(base_val, np.ndarray):
            base_val = float(base_val[0])
            
        contributions = pd.DataFrame({
            "Feature": self.feature_names,
            "Feature_Value": [single_row[f] for f in self.feature_names],
            "SHAP_Impact_kWh": np.round(shap_vals, 3)
        }).sort_values(by="SHAP_Impact_kWh", key=abs, ascending=False).reset_index(drop=True)
        
        pos_drivers = contributions[contributions["SHAP_Impact_kWh"] > 0].head(5)
        neg_drivers = contributions[contributions["SHAP_Impact_kWh"] < 0].head(5)
        
        return {
            "base_value_kwh": round(float(base_val), 2),
            "total_prediction_impact": round(float(np.sum(shap_vals)), 2),
            "top_positive_drivers": pos_drivers.to_dict(orient="records"),
            "top_negative_drivers": neg_drivers.to_dict(orient="records"),
            "all_contributions": contributions
        }
