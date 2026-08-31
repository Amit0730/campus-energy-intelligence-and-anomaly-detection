"""
Campus Energy Dataset Generator
Generates realistic multi-building hourly energy consumption data with weather,
operational calendar, occupancy profiles, and realistic anomaly events.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple


BUILDING_PROFILES = {
    "Academic_Block_1": {
        "area_sqm": 25000,
        "base_load_kw": 45.0,
        "peak_multiplier": 4.5,
        "hvac_sensitivity": 0.45,
        "type": "Academic",
        "primary_hours": (8, 18),
        "weekend_factor": 0.15,
        "vacation_factor": 0.20,
        "exam_factor": 1.25,
    },
    "Central_Library": {
        "area_sqm": 12000,
        "base_load_kw": 30.0,
        "peak_multiplier": 3.2,
        "hvac_sensitivity": 0.35,
        "type": "Library",
        "primary_hours": (7, 23),
        "weekend_factor": 0.75,
        "vacation_factor": 0.40,
        "exam_factor": 1.65,
    },
    "Hostel_Girls_Block": {
        "area_sqm": 18000,
        "base_load_kw": 25.0,
        "peak_multiplier": 3.0,
        "hvac_sensitivity": 0.25,
        "type": "Residential",
        "primary_hours": (17, 24),
        "weekend_factor": 1.30,
        "vacation_factor": 0.15,
        "exam_factor": 1.35,
    },
    "Hostel_Boys_Block": {
        "area_sqm": 22000,
        "base_load_kw": 32.0,
        "peak_multiplier": 3.1,
        "hvac_sensitivity": 0.28,
        "type": "Residential",
        "primary_hours": (17, 24),
        "weekend_factor": 1.35,
        "vacation_factor": 0.15,
        "exam_factor": 1.30,
    },
    "Dining_Hall_Mess": {
        "area_sqm": 8000,
        "base_load_kw": 20.0,
        "peak_multiplier": 5.0,
        "hvac_sensitivity": 0.30,
        "type": "Commercial",
        "primary_hours": (6, 22),
        "weekend_factor": 1.10,
        "vacation_factor": 0.10,
        "exam_factor": 1.15,
    }
}


def generate_weather_series(date_range: pd.DatetimeIndex, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic hourly realistic weather data."""
    rng = np.random.default_rng(seed)
    n = len(date_range)
    
    day_of_year = date_range.dayofyear.values
    hour = date_range.hour.values
    
    # Seasonal yearly temperature curve (Northern Hemisphere campus climate)
    # Peak summer in May-June (~38-42 C), mild winter in Jan (~10-18 C)
    seasonal_temp = 26.0 + 12.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.25)
    # Diurnal temperature cycle: coolest at 5am, warmest at 3pm (15:00)
    diurnal_temp = 5.5 * np.sin(2 * np.pi * (hour - 9) / 24)
    weather_noise = rng.normal(0, 1.8, size=n)
    temperature = np.clip(seasonal_temp + diurnal_temp + weather_noise, 4.0, 46.0)
    
    # Humidity is negatively correlated with temperature, with monsoon boost in Jul-Aug
    monsoon_boost = 15.0 * np.exp(-0.5 * ((day_of_year - 210) / 30) ** 2)
    humidity = 70.0 - 0.7 * (temperature - 25.0) + monsoon_boost + rng.normal(0, 4.0, size=n)
    humidity = np.clip(humidity, 18.0, 98.0)
    
    # Solar Radiation (W/m2) - 0 at night, peaks at solar noon (hour 12-13)
    solar_base = np.maximum(0, np.sin(np.pi * (hour - 6) / 12))
    cloud_cover = rng.uniform(0.7, 1.0, size=n)
    solar_radiation = 950.0 * solar_base * cloud_cover * np.where((hour >= 6) & (hour <= 18), 1.0, 0.0)
    
    # Wind Speed (m/s)
    wind_speed = np.clip(rng.weibull(2.0, size=n) * 3.5 + 0.5 * np.sin(2 * np.pi * hour / 24), 0.5, 18.0)
    
    return pd.DataFrame({
        "timestamp": date_range,
        "temperature_c": np.round(temperature, 2),
        "humidity_pct": np.round(humidity, 2),
        "solar_radiation_wm2": np.round(solar_radiation, 2),
        "wind_speed_ms": np.round(wind_speed, 2)
    })


def generate_campus_dataset(
    start_date: str = "2024-01-01 00:00:00",
    end_date: str = "2025-12-31 23:00:00",
    seed: int = 42,
    anomaly_rate: float = 0.025
) -> pd.DataFrame:
    """Generates complete multi-building campus hourly dataset with anomalies."""
    rng = np.random.default_rng(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="h")
    
    weather_df = generate_weather_series(date_range, seed=seed)
    
    all_building_records = []
    
    for building_name, profile in BUILDING_PROFILES.items():
        b_df = weather_df.copy()
        b_df["building_id"] = building_name
        b_df["building_type"] = profile["type"]
        b_df["area_sqm"] = profile["area_sqm"]
        
        # Calendar features
        timestamps = b_df["timestamp"]
        b_df["hour"] = timestamps.dt.hour
        b_df["day_of_week"] = timestamps.dt.dayofweek
        b_df["day_name"] = timestamps.dt.day_name()
        b_df["month"] = timestamps.dt.month
        b_df["is_weekend"] = (b_df["day_of_week"] >= 5).astype(int)
        
        # Campus academic calendar logic
        # Exam weeks: mid-term (early March, early Oct) & end-term (early Dec, early May)
        day_of_year = timestamps.dt.dayofyear
        is_exam = (
            ((day_of_year >= 65) & (day_of_year <= 75)) |    # Spring Mid-term
            ((day_of_year >= 125) & (day_of_year <= 140)) |  # Spring End-term
            ((day_of_year >= 275) & (day_of_year <= 285)) |  # Fall Mid-term
            ((day_of_year >= 335) & (day_of_year <= 350))    # Fall End-term
        ).astype(int)
        b_df["is_exam_week"] = is_exam
        
        # Vacation periods: Summer break (June 1 - July 20), Winter break (Dec 20 - Jan 5)
        is_vacation = (
            ((day_of_year >= 152) & (day_of_year <= 201)) |  # Summer
            ((day_of_year >= 354) | (day_of_year <= 5))       # Winter
        ).astype(int)
        b_df["is_vacation"] = is_vacation
        
        # Official holidays (sample major national/campus holidays)
        holidays_doy = [26, 75, 105, 227, 275, 305, 359]
        b_df["is_holiday"] = day_of_year.isin(holidays_doy).astype(int)
        
        # Base Occupancy Model
        occupancy = np.zeros(len(b_df))
        h = b_df["hour"].values
        dow = b_df["day_of_week"].values
        is_wk = b_df["is_weekend"].values
        is_vac = b_df["is_vacation"].values
        is_ex = b_df["is_exam_week"].values
        
        if profile["type"] == "Academic":
            # High 9am - 5pm, minimal at night and weekends
            base_occ = np.where((h >= 8) & (h <= 17), np.sin(np.pi * (h - 8) / 9) * 0.9, 0.05)
            occupancy = base_occ * np.where(is_wk == 1, 0.10, 1.0) * np.where(is_vac == 1, 0.15, 1.0)
            occupancy = np.where(is_ex == 1, occupancy * 1.15, occupancy)
        elif profile["type"] == "Library":
            # Steady daytime with late night boost
            base_occ = np.where((h >= 7) & (h <= 23), 0.45 + 0.40 * np.sin(np.pi * (h - 7) / 16), 0.08)
            occupancy = base_occ * np.where(is_vac == 1, 0.35, 1.0)
            occupancy = np.where(is_ex == 1, np.minimum(1.0, occupancy * 1.5), occupancy)
        elif profile["type"] == "Residential":
            # Hostel: peaks early morning (7-9am) and evening/night (18-24), high on weekends
            base_occ = np.where((h >= 18) | (h <= 8), 0.85, 0.25)
            occupancy = base_occ * np.where(is_wk == 1, 1.2, 1.0) * np.where(is_vac == 1, 0.20, 1.0)
        elif profile["type"] == "Commercial":
            # Mess: tri-modal curve (Breakfast 7-9, Lunch 12-14, Dinner 19-21)
            b_peak = np.exp(-0.5 * ((h - 8.0) / 1.0) ** 2) * 0.85
            l_peak = np.exp(-0.5 * ((h - 13.0) / 1.2) ** 2) * 0.95
            d_peak = np.exp(-0.5 * ((h - 20.0) / 1.2) ** 2) * 0.90
            occupancy = np.clip(b_peak + l_peak + d_peak + 0.05, 0.02, 1.0) * np.where(is_vac == 1, 0.15, 1.0)
        
        occupancy = np.clip(occupancy + rng.normal(0, 0.04, size=len(b_df)), 0.01, 1.0)
        b_df["occupancy_rate"] = np.round(occupancy, 3)
        
        # Energy Component Modeling (Physics-based empirical formulas)
        # 1. HVAC Load (strongly dependent on temperature above 22C cooling setpoint or below 16C heating)
        cooling_deg = np.maximum(0, b_df["temperature_c"].values - 22.0)
        heating_deg = np.maximum(0, 16.0 - b_df["temperature_c"].values)
        thermal_load = cooling_deg * 1.2 + heating_deg * 0.8
        
        hvac_base = profile["base_load_kw"] * profile["hvac_sensitivity"]
        hvac_kwh = (
            hvac_base +
            thermal_load * (profile["area_sqm"] / 1000.0) * 0.45 * (0.3 + 0.7 * b_df["occupancy_rate"].values) +
            (b_df["solar_radiation_wm2"].values / 1000.0) * 4.0
        )
        
        # 2. Lighting Load (dependent on occupancy, darkness, area)
        daylight_available = b_df["solar_radiation_wm2"].values > 100
        lighting_multiplier = np.where(daylight_available, 0.4, 1.0)
        lighting_kwh = (
            profile["base_load_kw"] * 0.25 +
            b_df["occupancy_rate"].values * (profile["area_sqm"] / 1000.0) * 1.8 * lighting_multiplier
        )
        
        # 3. Equipment & Plug Loads
        equipment_kwh = (
            profile["base_load_kw"] * 0.30 +
            b_df["occupancy_rate"].values * (profile["area_sqm"] / 1000.0) * 2.2 +
            rng.normal(0, 1.0, size=len(b_df))
        )
        
        # Clean total power calculation
        total_power = hvac_kwh + lighting_kwh + equipment_kwh + rng.normal(0, 1.5, size=len(b_df))
        total_power = np.clip(total_power, profile["base_load_kw"] * 0.5, None)
        
        b_df["hvac_power_kwh"] = np.round(np.maximum(1.0, hvac_kwh), 2)
        b_df["lighting_power_kwh"] = np.round(np.maximum(0.5, lighting_kwh), 2)
        b_df["equipment_power_kwh"] = np.round(np.maximum(0.5, equipment_kwh), 2)
        b_df["total_power_kwh"] = np.round(total_power, 2)
        
        # Inject realistic labeled anomalies
        b_df["is_anomaly"] = 0
        b_df["anomaly_type"] = "Normal"
        b_df["anomaly_severity"] = "None"
        b_df["expected_normal_kwh"] = b_df["total_power_kwh"]
        
        # Anomaly Injection Logic
        num_anomalies = int(len(b_df) * anomaly_rate)
        candidate_indices = rng.choice(len(b_df), size=num_anomalies, replace=False)
        
        for idx in candidate_indices:
            row_h = b_df.iloc[idx]["hour"]
            row_wk = b_df.iloc[idx]["is_weekend"]
            dice = rng.uniform()
            
            if dice < 0.35:
                # Anomaly Scenario 1: Overnight HVAC Leak / Equipment Left ON
                # Spikes off-hours power dramatically
                surge_amount = float(profile["base_load_kw"] * rng.uniform(2.5, 4.2))
                b_df.iat[idx, b_df.columns.get_loc("is_anomaly")] = 1
                b_df.iat[idx, b_df.columns.get_loc("anomaly_type")] = "Off-Hours HVAC Leakage"
                b_df.iat[idx, b_df.columns.get_loc("anomaly_severity")] = "High"
                b_df.iat[idx, b_df.columns.get_loc("hvac_power_kwh")] += surge_amount * 0.7
                b_df.iat[idx, b_df.columns.get_loc("total_power_kwh")] += surge_amount
                
            elif dice < 0.65:
                # Anomaly Scenario 2: Severe Equipment Short/Overload or Unmetered Campus Event
                surge_amount = float(b_df.iat[idx, b_df.columns.get_loc("total_power_kwh")] * rng.uniform(0.6, 1.3))
                b_df.iat[idx, b_df.columns.get_loc("is_anomaly")] = 1
                b_df.iat[idx, b_df.columns.get_loc("anomaly_type")] = "Heavy Equipment Overload"
                b_df.iat[idx, b_df.columns.get_loc("anomaly_severity")] = "Critical"
                b_df.iat[idx, b_df.columns.get_loc("equipment_power_kwh")] += surge_amount
                b_df.iat[idx, b_df.columns.get_loc("total_power_kwh")] += surge_amount
                
            elif dice < 0.85:
                # Anomaly Scenario 3: Sensor Fault / Partial Substation Drop during daytime
                drop_factor = rng.uniform(0.25, 0.45)
                b_df.iat[idx, b_df.columns.get_loc("is_anomaly")] = 1
                b_df.iat[idx, b_df.columns.get_loc("anomaly_type")] = "Power Phase Loss / Sensor Fault"
                b_df.iat[idx, b_df.columns.get_loc("anomaly_severity")] = "Medium"
                b_df.iat[idx, b_df.columns.get_loc("total_power_kwh")] *= drop_factor
                
            else:
                # Anomaly Scenario 4: Basal Energy Creep / Lighting Relay Jammed
                surge_amount = float(profile["base_load_kw"] * rng.uniform(0.8, 1.5))
                b_df.iat[idx, b_df.columns.get_loc("is_anomaly")] = 1
                b_df.iat[idx, b_df.columns.get_loc("anomaly_type")] = "Lighting Relay Jam / Basal Creep"
                b_df.iat[idx, b_df.columns.get_loc("anomaly_severity")] = "Low"
                b_df.iat[idx, b_df.columns.get_loc("lighting_power_kwh")] += surge_amount
                b_df.iat[idx, b_df.columns.get_loc("total_power_kwh")] += surge_amount

        all_building_records.append(b_df)
        
    full_df = pd.concat(all_building_records, ignore_index=True)
    full_df.sort_values(by=["timestamp", "building_id"], inplace=True)
    full_df.reset_index(drop=True, inplace=True)
    return full_df


if __name__ == "__main__":
    df = generate_campus_dataset()
    print(f"Generated {len(df)} records across {df['building_id'].nunique()} buildings.")
    print(df.head())
    print("\nAnomaly distribution:")
    print(df["anomaly_type"].value_counts())
