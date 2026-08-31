# ⚡ Campus Energy Consumption Intelligence & Anomaly Detection
### INT395: Supervised Learning — Comprehensive Course Project (Project 15)
**Total Evaluation Marks**: 35 Marks | **Team Size**: 3 Students

---

## 📌 1. Project Overview & Problem Definition

### 1.1 Real-World Campus Challenge
Modern university campuses encompass diverse facility types (lecture complexes, research laboratories, high-density residential hostels, air-conditioned libraries, and commercial dining facilities) operating under dynamic academic calendars and fluctuating meteorological conditions. Unmonitored energy consumption leads to:
1. **Severe Energy Wastage**: HVAC systems and lighting left operating at full load during off-hours, holidays, or in empty buildings.
2. **Cost Overruns & Grid Demand Penalties**: Unpredicted peak loads exceeding contracted maximum demand limits with municipal utility providers.
3. **Equipment Breakdown & Safety Risks**: Unnoticed heavy equipment overloads, short-circuits, and phase imbalances.
4. **Carbon Emissions**: Avoidable electricity consumption directly undermines university net-zero sustainability commitments.

### 1.2 Target Stakeholders
- **Campus Facility & Electrical Engineering Managers**: Real-time load monitoring, automated abnormal usage alerts, and preventive maintenance scheduling.
- **University Sustainability & Administration Office**: Energy Use Intensity (EUI) benchmarking, carbon reduction compliance, and budget planning.
- **Hostel Wardens & Academic Block Supervisors**: Automated notifications of off-hours equipment runaways and occupancy-linked energy leaks.

### 1.3 Machine Learning Objectives
- **Supervised Multi-Horizon Forecasting**: Predict building-level and campus-wide hourly electricity consumption ($kWh$) with $R^2 > 0.90$ and low MAPE using meteorological, temporal, and historical lag indicators.
- **Unsupervised Anomaly Detection & Risk Scoring**: Identify statistically significant consumption surges, off-hours leaks, sensor telemetry dropouts, and categorize them into actionable severity tiers (Low, Medium, High, Critical).
- **Explainable AI (XAI)**: Compute SHAP values to explain global campus energy drivers and decompose individual anomalous predictions into specific physical factors.
- **Operational Decision Support**: Interactive what-if scenario simulations for heatwaves, exam schedules, and energy conservation policies.

---

## 🏛️ 2. Campus Facilities & Multi-Building Scope

| Facility Name | Category | Floor Area ($m^2$) | Primary Operational Hours | Key Energy Drivers |
|---|---|---|---|---|
| **Academic Block 1** | Academic | 25,000 | 08:00 – 18:00 (Weekdays) | Lecture hall lighting, computer labs, central HVAC |
| **Central Library** | Academic / Study | 12,000 | 07:00 – 23:00 (7 Days/Week) | Reading hall HVAC, 24/7 server room, exam surges |
| **Hostel Girls Block** | Residential | 18,000 | 17:00 – 08:00 (Evenings/Weekends) | Personal appliances, water geysers, room cooling |
| **Hostel Boys Block** | Residential | 22,000 | 17:00 – 08:00 (Evenings/Weekends) | Resident electronics, evening study loads, weekends |
| **Dining Hall / Mess** | Commercial | 8,000 | Meal Surges (Breakfast, Lunch, Dinner) | Commercial refrigeration, kitchen ventilation, induction |

---

## 🔄 3. Complete End-to-End ML Pipeline Architecture

```
                                  [ RAW SENSOR & WEATHER DATA ]
                                                │
                                                ▼
                             [ PREPROCESSING & FEATURE ENGINEERING ]
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
             [ Cyclical & Thermal ]                            [ Time-Series Lags ]
        (sin/cos hour, CDH, Heat Index)                  (t-1, t-2, t-24, t-168, Rolling)
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                             [ CHRONOLOGICAL TRAIN / VAL / TEST SPLIT ]
                                                │
                                                ▼
                             [ SUPERVISED MODEL BENCHMARK ZOO ]
       ┌────────────────────┬───────────────────┬───────────────────┬──────────────────┐
       ▼                    ▼                   ▼                   ▼                  ▼
 [ Persistence ]      [ Ridge Reg ]       [ Random Forest ]   [ LightGBM / XGB ]  [ Stacking Ens ]
       └────────────────────┴───────────────────┬───────────────────┴──────────────────┘
                                                │
                                                ▼
                             [ HYPERPARAMETER TUNING & EVALUATION ]
                                 (R², RMSE, MAE, MAPE, Dir-Acc)
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
          [ EXPLAINABLE AI (SHAP) ]                      [ ANOMALY DETECTION ENGINE ]
       (Global & Local Waterfall XAI)                  (Isolation Forest + Residuals)
                       │                                                 │
                       └────────────────────────┬────────────────────────┘
                                                │
                                                ▼
                             [ INTERACTIVE STREAMLIT DEPLOYMENT ]
```

---

## ⚙️ 4. Feature Engineering Methodology

1. **Temporal & Cyclical Transformations**:
   - Hour encoding: $\sin(2\pi \cdot \text{hour} / 24)$, $\cos(2\pi \cdot \text{hour} / 24)$
   - Day of week encoding: $\sin(2\pi \cdot \text{dow} / 7)$, $\cos(2\pi \cdot \text{dow} / 7)$
   - Month encoding: $\sin(2\pi \cdot (\text{month}-1) / 12)$, $\cos(2\pi \cdot (\text{month}-1) / 12)$
   - Academic flags: `is_exam_week`, `is_vacation`, `is_holiday`, `is_business_hour`

2. **Thermodynamic & Environmental Indices**:
   - **Cooling Degree Hours (CDH)**: $\text{CDH} = \max(0, T_{\text{outdoor}} - 22.0^\circ\text{C})$
   - **Heating Degree Hours (HDH)**: $\text{HDH} = \max(0, 16.0^\circ\text{C} - T_{\text{outdoor}})$
   - **Heat Index (Rothfusz equation)**: Non-linear combination of temperature and relative humidity.
   - **Solar Thermal Load**: $\text{Solar Radiation} \times \text{CDH} / 1000$

3. **Time-Series Autoregressive Lag & Rolling Statistics**:
   - Short-term lags: $t-1\text{h}, t-2\text{h}$
   - Daily seasonal lag: $t-24\text{h}, t-48\text{h}$
   - Weekly seasonal lag: $t-168\text{h}$
   - Rolling aggregates: Historical $6\text{h}, 24\text{h}, 168\text{h}$ mean, standard deviation, min, and max calculated strictly backwards to prevent future lookahead leakage.

---

## 📊 5. Supervised Model Zoo & Comparative Evaluation

| Model Algorithm | Paradigm / Family | Optimization Strategy |
|---|---|---|
| **Persistence Baseline** | Heuristic Benchmark | Lag-24h persistence fallback |
| **Ridge Regression** | L2-Regularized Linear | StandardScaler + L2 penalty $\alpha=10.0$ |
| **Random Forest Regressor** | Bagging Ensemble | 100 trees, max depth 16, feature subsampling |
| **LightGBM Regressor** | Histogram GBDT | 200 estimators, leaf-wise tree growth, lr=0.05 |
| **XGBoost Regressor** | Extreme Gradient Boosting | 200 estimators, depth-wise regularization, lr=0.05 |
| **Stacking Ensemble** | Heterogeneous Stacking | Ridge meta-learner blending Ridge, RF, LGBM, XGB |

### Standard Regression Evaluation Metrics Computed:
- **Root Mean Squared Error (RMSE)**: $\sqrt{\frac{1}{N} \sum (y_i - \hat{y}_i)^2}$
- **Mean Absolute Error (MAE)**: $\frac{1}{N} \sum |y_i - \hat{y}_i|$
- **Mean Absolute Percentage Error (MAPE %)**: $\frac{100\%}{N} \sum \left|\frac{y_i - \hat{y}_i}{y_i}\right|$
- **Coefficient of Determination ($R^2$)**: $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Directional Accuracy (%)**: Agreement of hour-to-hour trend slope signs.

---

## ⚠️ 6. Anomaly Detection & Wastage Risk Engine

1. **Isolation Forest**: Identifies multi-dimensional structural anomalies in feature space $(\text{Power}, \text{Temp}, \text{Occupancy}, \text{Hour}, \text{CDH})$.
2. **Studentized Residual Outlier Detector**: Dynamic $\pm 3\sigma$ prediction confidence intervals from the best supervised forecaster.
3. **Root Cause Diagnostics**:
   - *Off-Hours Energy Leakage*: Overnight consumption $> 200\%$ baseline with $0\%$ occupancy.
   - *Heavy Equipment Overload*: Sudden positive spike ($> +3.5\sigma$) uncorrelated with weather.
   - *Sensor Telemetry / Phase Loss*: Sudden negative drop ($< -3.0\sigma$) during occupied hours.
   - *Thermal Inefficiency*: Excessive HVAC draw during extreme ambient temperatures.
4. **Financial & Environmental Quantifier**:
   $$\text{Wasted Energy (kWh)} = \max(0, y_{\text{actual}} - \hat{y}_{\text{expected}})$$
   $$\text{Financial Loss (₹)} = \text{Wasted Energy} \times \text{Tariff (₹ 8.5/kWh)}$$
   $$\text{Excess Carbon (kg CO}_2\text{)} = \text{Wasted Energy} \times 0.82\text{ kg CO}_2\text{/kWh}$$

---

## 🧠 7. Explainable AI (SHAP)
- **Global SHAP Summary**: Identifies top influential features across the entire campus grid.
- **Local Waterfall Explanations**: Decomposes any selected individual hourly forecast:
  $$\hat{y} = E[y] + \sum_{j=1}^{M} \phi_j$$
  where $E[y]$ is the base value (campus mean load) and $\phi_j$ is the exact additive contribution of feature $j$.

---

## 🚀 8. Project Structure & Organization

```
superviesd learning/
│
├── data/
│   ├── raw_campus_energy_data.csv          # 2-year hourly raw dataset
│   ├── featured_campus_energy_data.csv     # Engineered features dataset
│   ├── test_predictions_with_anomalies.csv # Test evaluation + anomaly labels
│   ├── shap_feature_importance.csv         # Global SHAP importance table
│   └── metrics_summary.json                # Complete benchmark JSON
│
├── notebooks/
│   └── Campus_Energy_Intelligence_EDA_Modeling.ipynb  # Step-by-step notebook
│
├── saved_models/
│   ├── campus_energy_models.joblib         # Serialized ML model zoo
│   └── energy_anomaly_detector.joblib      # Serialized Isolation Forest
│
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── data_generator.py                   # Realistic campus dataset generator
│   ├── preprocessing.py                    # Feature engineering & splits
│   ├── models.py                           # Model implementations & zoo
│   ├── anomaly_detector.py                 # Hybrid anomaly engine & diagnostics
│   ├── explainability.py                   # SHAP TreeExplainer wrapper
│   └── pipeline.py                         # End-to-end training manager
│
├── app.py                                  # Streamlit Interactive Web Application
├── train.py                                # CLI Automated Pipeline Runner
├── requirements.txt                        # Python dependencies
├── VIVA_PREPARATION_GUIDE.md               # 3-Student Viva & Rubric Defence Guide
└── README.md                               # Comprehensive Project Documentation
```

---

## 💻 9. Installation & Quick Start Guide

### Prerequisites
- Python 3.10 or 3.11 installed.

### Step 1: Clone Repository and Navigate to Folder
```bash
cd "c:\Users\amitk\Desktop\superviesd learning"
```

### Step 2: Set Up Virtual Environment & Install Dependencies
```bash
# Using standard venv
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Run the End-to-End Training Pipeline
```bash
python train.py
```
*This will generate the 2-year campus dataset, compute engineered features, train all 6 models, perform anomaly detection, compute SHAP values, and serialize all artifacts to `saved_models/`.*

### Step 4: Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to interact with the full dashboard.

---

## 👥 10. Team Responsibilities & Viva Division
- **Student 1**: Problem Formulation, Data Acquisition, Exploratory Data Analysis & Preprocessing Pipelines.
- **Student 2**: Supervised Model Zoo Development, Ensemble Learning, Hyperparameter Tuning & Cross-Validation.
- **Student 3**: Anomaly Detection, Explainable AI (SHAP), Streamlit Dashboard Deployment & What-If Simulation.
