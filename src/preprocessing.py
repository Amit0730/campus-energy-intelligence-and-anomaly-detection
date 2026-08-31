"""
Preprocessing & Feature Engineering Pipeline
Constructs time-series lag features, cyclic transforms, weather indices,
and handles train/val/test chronological splitting without data leakage.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def compute_heat_index(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Computes simplified Heat Index (°C)."""
    t_f = temp_c * 9.0 / 5.0 + 32.0
    hi_f = (
        -42.379 +
        2.04901523 * t_f +
        10.14333127 * rh_pct -
        0.22475541 * t_f * rh_pct -
        0.00683783 * t_f * t_f -
        0.05481717 * rh_pct * rh_pct +
        0.00122874 * t_f * t_f * rh_pct +
        0.00085282 * t_f * rh_pct * rh_pct -
        0.00000199 * t_f * t_f * rh_pct * rh_pct
    )
    hi_c = (hi_f - 32.0) * 5.0 / 9.0
    return np.where(temp_c >= 26.0, hi_c, temp_c)


def engineer_features(df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
    """
    Applies comprehensive feature engineering to campus energy dataset.
    Computes lag features, rolling statistics per building, cyclical features,
    and domain-specific energy indicators.
    """
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data.sort_values(by=["building_id", "timestamp"], inplace=True)
    data.reset_index(drop=True, inplace=True)
    
    # 1. Temporal & Cyclical Trigonometric Features
    hour = data["hour"].values
    dow = data["day_of_week"].values
    month = data["month"].values
    
    data["hour_sin"] = np.round(np.sin(2 * np.pi * hour / 24.0), 4)
    data["hour_cos"] = np.round(np.cos(2 * np.pi * hour / 24.0), 4)
    data["dow_sin"] = np.round(np.sin(2 * np.pi * dow / 7.0), 4)
    data["dow_cos"] = np.round(np.cos(2 * np.pi * dow / 7.0), 4)
    data["month_sin"] = np.round(np.sin(2 * np.pi * (month - 1) / 12.0), 4)
    data["month_cos"] = np.round(np.cos(2 * np.pi * (month - 1) / 12.0), 4)
    
    # Business / Active campus hour flag
    data["is_business_hour"] = (
        (data["hour"] >= 8) & (data["hour"] <= 18) & (data["is_weekend"] == 0)
    ).astype(int)
    
    # 2. Weather & Thermal Indices
    temp = data["temperature_c"].values
    rh = data["humidity_pct"].values
    solar = data["solar_radiation_wm2"].values
    wind = data["wind_speed_ms"].values
    
    data["cooling_degree_hours"] = np.round(np.maximum(0.0, temp - 22.0), 3)
    data["heating_degree_hours"] = np.round(np.maximum(0.0, 16.0 - temp), 3)
    data["heat_index_c"] = np.round(compute_heat_index(temp, rh), 2)
    data["solar_thermal_index"] = np.round(solar * data["cooling_degree_hours"] / 1000.0, 4)
    data["apparent_temperature_c"] = np.round(
        temp + 0.33 * (rh / 100.0 * 6.105 * np.exp(17.27 * temp / (237.7 + temp))) - 0.70 * wind - 4.0,
        2
    )
    
    # 3. Occupancy Interaction Terms
    data["occ_x_temp"] = np.round(data["occupancy_rate"] * data["temperature_c"], 3)
    data["occ_x_exam"] = np.round(data["occupancy_rate"] * data["is_exam_week"], 3)
    data["occ_x_business"] = np.round(data["occupancy_rate"] * data["is_business_hour"], 3)
    
    # 4. Lag Features & Rolling Window Aggregations (Computed per Building)
    grouped = data.groupby("building_id")
    
    # Lags (1 hour, 2 hours, 24 hours, 48 hours, 168 hours = 1 week)
    lags = [1, 2, 24, 48, 168]
    for lag in lags:
        data[f"power_lag_{lag}h"] = grouped["total_power_kwh"].shift(lag)
        data[f"temp_lag_{lag}h"] = grouped["temperature_c"].shift(lag)
        
    # Rolling Statistics over past historical windows (using closed='left' equivalent via shift)
    for window in [6, 24, 168]:
        shifted_power = grouped["total_power_kwh"].shift(1)
        data[f"power_rolling_mean_{window}h"] = shifted_power.groupby(data["building_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        data[f"power_rolling_std_{window}h"] = shifted_power.groupby(data["building_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).std().fillna(0.0)
        )
        data[f"power_rolling_max_{window}h"] = shifted_power.groupby(data["building_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).max()
        )
        data[f"power_rolling_min_{window}h"] = shifted_power.groupby(data["building_id"]).transform(
            lambda s: s.rolling(window, min_periods=1).min()
        )

    # Fill earliest warm-up NaNs with forward/backward fill within building
    lag_cols = [c for c in data.columns if "lag" in c or "rolling" in c]
    data[lag_cols] = data.groupby("building_id")[lag_cols].bfill().ffill()
    
    # Round numerical features
    for c in lag_cols:
        data[c] = np.round(data[c], 3)
        
    return data


def get_feature_columns() -> Tuple[List[str], str]:
    """Returns list of predictive feature column names and target column name."""
    features = [
        # Building context
        "area_sqm",
        # Calendar & Occupancy
        "hour", "day_of_week", "month", "is_weekend", "is_exam_week",
        "is_vacation", "is_holiday", "is_business_hour", "occupancy_rate",
        # Cyclical transforms
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        # Weather & Thermal
        "temperature_c", "humidity_pct", "solar_radiation_wm2", "wind_speed_ms",
        "cooling_degree_hours", "heating_degree_hours", "heat_index_c",
        "solar_thermal_index", "apparent_temperature_c",
        # Interactions
        "occ_x_temp", "occ_x_exam", "occ_x_business",
        # Lags
        "power_lag_1h", "power_lag_2h", "power_lag_24h", "power_lag_48h", "power_lag_168h",
        "temp_lag_1h", "temp_lag_24h",
        # Rolling stats
        "power_rolling_mean_6h", "power_rolling_std_6h",
        "power_rolling_mean_24h", "power_rolling_std_24h", "power_rolling_max_24h", "power_rolling_min_24h",
        "power_rolling_mean_168h", "power_rolling_std_168h"
    ]
    target = "total_power_kwh"
    return features, target


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/validation/test split per building to prevent data leakage.
    """
    train_dfs, val_dfs, test_dfs = [], [], []
    
    for _, b_group in df.groupby("building_id"):
        b_group_sorted = b_group.sort_values(by="timestamp").reset_index(drop=True)
        n = len(b_group_sorted)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        train_dfs.append(b_group_sorted.iloc[:train_end])
        val_dfs.append(b_group_sorted.iloc[train_end:val_end])
        test_dfs.append(b_group_sorted.iloc[val_end:])
        
    train_df = pd.concat(train_dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    val_df = pd.concat(val_dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    test_df = pd.concat(test_dfs, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    
    return train_df, val_df, test_df
