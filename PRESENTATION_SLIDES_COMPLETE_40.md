# 📊 PROJECT 15: CAMPUS ENERGY CONSUMPTION INTELLIGENCE & ANOMALY DETECTION
## Complete 40-Slide Presentation Deck with Experimental Results
**Course**: INT395 — Supervised Learning  
**Project Number**: 15  
**Course Weightage**: 35 Marks  

---

### Slide 1 — Title Slide
# Campus Energy Consumption Intelligence & Anomaly Detection
### *An End-to-End Machine Learning Platform for Multi-Building Load Forecasting, Anomaly Risk Scoring, and Operational Interpretability*

**Presented by:**
- **Student 1**: [Name] — Roll No. [XXXXXXX] *(Data Acquisition, Preprocessing & EDA Lead)*
- **Student 2**: [Name] — Roll No. [XXXXXXX] *(Forecasting Models, Ensembles & Tuning Lead)*
- **Student 3**: [Name] — Roll No. [XXXXXXX] *(Anomaly Detection, SHAP XAI & Deployment Lead)*

**Department of Computer Science & Engineering**  
**Course**: INT395 — Supervised Learning (CA Project — 35 Marks)

---

### Slide 2 — Introduction: Campus Energy Landscape
#### Why Campus Energy Intelligence Matters
Modern university campuses function as small self-contained micro-cities with distinct functional zones:
- **Academic Complex & Classrooms**: High daytime demand (08:00–18:00), low off-hours baseline.
- **24/7 Central Library**: High continuous HVAC and lighting load, major spikes during exam weeks.
- **Student Residential Hostels**: Morning (06:30–09:00) and evening/night (18:00–01:00) peaks, heavy weekend load.
- **Dining Halls / Mess**: Tri-modal surge curve aligned with meal preparation schedules.
- **Laboratories & Computing Centers**: Sensitive power-quality loads and high baseline equipment draw.

#### Dynamic Drivers of Campus Energy Consumption
Energy consumption is highly non-linear and driven by:
1. **Diurnal Human Behavior & Schedules** (Classes, meal timings, exam weeks, vacations).
2. **Meteorological Extremes** (Ambient temperature, solar radiation, humidity thermal stress).
3. **Operational Discipline** (Equipment left operating unattended overnight).

---

### Slide 3 — Problem Statement
#### The Existing Campus Energy Crisis
Current campus energy monitoring relies on manual meter logging or retrospective monthly utility bills. This legacy paradigm leads to:
1. **Unnoticed Off-Hours Energy Leaks**: Chوتی air-conditioning units and lighting left on overnight in unoccupied lecture halls.
2. **Unanticipated Peak-Demand Penalties**: Exceeding sanctioned maximum contract demand with electricity supply boards incurring heavy financial penalties.
3. **Equipment Overloads & Failures**: Lab equipment short-circuits or transformer phase imbalances going undetected until catastrophic breakdown.
4. **Lack of Predictive Foresight**: Facility managers have no scientific methodology to anticipate energy budget demands for upcoming heatwaves or academic festivals.

#### Core Problem Formulation
> *"How can supervised machine learning and unsupervised outlier detection be synergized to forecast multi-building campus electricity load ($kWh$), establish empirical normal operating baselines, detect energy anomalies with calibrated severity scoring, and deliver interpretable root-cause insights via a deployed decision support platform?"*

---

### Slide 4 — Project Objectives & ML Scope
#### Primary Objective
To design, train, evaluate, and deploy an end-to-end Machine Learning intelligence platform for university campus electricity management.

#### Specific Technical Objectives
1. **Data Acquisition & Modeling**: Model 2 full years (87,720 hourly records) of campus multi-building operations with environmental and academic calendar metrics.
2. **Exploratory Data Analysis**: Uncover diurnal, weekly, seasonal, and thermal correlation patterns.
3. **Feature Engineering**: Formulate 43 predictive features (cyclical trigonometric transforms, thermodynamic degree-hours, autoregressive lags, and rolling aggregates) with strict zero-leakage temporal splitting.
4. **Supervised Model Zoo**: Benchmark 6 diverse algorithms (Persistence Baseline, Ridge Regression, Random Forest, LightGBM, XGBoost, Stacking Ensemble).
5. **Hyperparameter Optimization**: Tune gradient boosted trees with `TimeSeriesSplit` cross-validation.
6. **Hybrid Anomaly Detection**: Formulate an Isolation Forest and studentized forecast residual engine ($z \ge 2.5\sigma$).
7. **Actionable Risk Scoring**: Compute a 0–100 Energy Risk Index, financial wastage ($₹$), and excess carbon ($kg CO_2$).
8. **Explainable AI (XAI)**: Decompose global drivers and local predictions using SHAP (SHapley Additive exPlanations).
9. **Interactive Deployment**: Deliver a 7-module Streamlit web application with What-If simulation capabilities.

---

### Slide 5 — Target Users & Institutional Stakeholders

```
┌──────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Stakeholder Persona              │ Operational Benefit from Platform                      │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Campus Electrical & Facility  │ Real-time load monitoring, automated anomaly alerts,   │
│    Engineers                     │ preventive maintenance & transformer load balancing.   │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 2. Energy & Sustainability Cell  │ Energy Use Intensity (EUI in kWh/m²/yr) tracking,      │
│                                  │ avoidable carbon emission accounting & ESG compliance. │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 3. Hostel Wardens & Building     │ Instant alerts for off-hours equipment runaways and    │
│    Supervisors                   │ unauthorized high-power appliances in dormitories.     │
├──────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 4. University Executive Board &  │ Predictive energy budget forecasting, tariff savings,  │
│    Chief Financial Officer       │ and solar microgrid capacity expansion planning.       │
└──────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

### Slide 6 — Proposed Dual-Track ML Solution Architecture

```
                                  [ MULTI-BUILDING SENSOR & WEATHER DATA ]
                                                     │
                                                     ▼
                                      [ DATA PREPROCESSING & EDA ]
                                                     │
                                                     ▼
                                [ 43-DIMENSIONAL FEATURE ENGINEERING ]
                                                     │
                                 ┌───────────────────┴───────────────────┐
                                 ▼                                       ▼
                       [ TRACK A: FORECASTING ]               [ TRACK B: ANOMALY DETECTION ]
                       Supervised Model Zoo                   Isolation Forest + Residuals
                       (Ridge, RF, XGB, LGBM, Stack)          (Outliers, Drops, HVAC Runaways)
                                 │                                       │
                                 ▼                                       ▼
                       Future Expected Load (kWh)              Anomaly Severity & Risk Score
                                 │                                       │
                                 └───────────────────┬───────────────────┘
                                                     │
                                                     ▼
                                      [ EXPLAINABLE AI (SHAP XAI) ]
                                                     │
                                                     ▼
                               [ DEPLOYED STREAMLIT COMMAND DASHBOARD ]
```

---

### Slide 7 — Dataset Acquisition & Attributes Description
The dataset captures hourly time-series operations over **2 full chronological years (2024-01-01 to 2025-12-31)** across 5 campus buildings:

| Feature Variable | Category | Unit / Format | Description & Physical Meaning |
|---|---|---|---|
| `timestamp` | Temporal | YYYY-MM-DD HH:MM | Hourly timestamp of measurement |
| `building_id` | Categorical | String | Unique facility identifier (5 campus buildings) |
| `building_type` | Categorical | String | Functional type (Academic, Library, Residential, Commercial) |
| `area_sqm` | Physical | $m^2$ | Total built-up floor area (8,000 to 25,000 $m^2$) |
| `temperature_c` | Meteorological | $^\circ\text{C}$ | Outdoor dry-bulb ambient temperature ($4.0^\circ\text{C} - 46.0^\circ\text{C}$) |
| `humidity_pct` | Meteorological | $\%$ | Relative humidity ($18.0\% - 98.0\%$) |
| `solar_radiation_wm2` | Meteorological | $W/m^2$ | Direct and diffuse solar irradiance ($0 - 950 W/m^2$) |
| `wind_speed_ms` | Meteorological | $m/s$ | Ambient wind velocity ($0.5 - 18.0 m/s$) |
| `occupancy_rate` | Behavioral | $0.0 - 1.0$ | Dynamic occupant density index |
| `is_weekend` / `is_holiday` | Calendar | Binary ($0/1$) | Weekend and official holiday indicators |
| `is_exam_week` / `vacation`| Academic | Binary ($0/1$) | Mid-term/End-term examination and semester break flags |
| `total_power_kwh` | **Target** | $kWh$ | **Total gross active electricity demand (Target Variable)** |

---

### Slide 8 — Dataset Characteristics & Quality Statistics
#### Concrete Dataset Dimensions
- **Total Processed Records**: **87,720 hourly records** (17,544 hours per building $\times$ 5 buildings)
- **Time Period**: January 1, 2024 00:00 to December 31, 2025 23:00 (730 continuous days)
- **Target Variable Range**: $10.0\text{ kWh}$ to $285.4\text{ kWh}$ (Campus Mean: $58.7\text{ kWh}$)
- **Missing / Null Values**: **0.0%** (Clean telemetry structure with backward/forward fill imputation)
- **Duplicate Records**: **0** (Verified unique `[timestamp, building_id]` index pairs)
- **Sampling Frequency**: Hourly continuous intervals ($\Delta t = 1.0\text{ hour}$)

---

### Slide 9 — EDA: Energy Consumption Distribution

```
  Frequency
      ▲
      │       Normal Operating Zone
      │         ┌───────────────┐
      │        ┌┘               └┐
      │       ┌┘                 └┐
      │      ┌┘                   └┐           Rare Severe Anomaly Spikes
      │     ┌┘                     └┐                   ┌──┐
      └─────┴───────────────────────┴───────────────────┴──┴────────►
           20       60       100      140      180     240   280  Electricity (kWh)
```

#### Analytical Interpretation:
- Over **88%** of hourly consumption values reside in the standard operating envelope ($25 - 110\text{ kWh}$).
- The distribution is moderately right-skewed with a long positive tail representing summer mid-day peak HVAC cooling surges and abnormal equipment overloads.

---

### Slide 10 — EDA: Time-Based Consumption Dynamics
#### 1. Diurnal 24-Hour Profile
- **Academic Complex**: Bell-shaped curve peaking at **14:00 (125 kWh)** and dropping to base standby load (**22 kWh**) between 22:00 and 06:00.
- **Hostel Blocks**: Dual-peak profile: morning departure (07:00–08:30) and evening return (18:00–23:30, **88 kWh**).
- **Dining Mess**: Tri-modal surges corresponding to Breakfast (08:00), Lunch (13:00), and Dinner (20:00).

#### 2. Weekly & Seasonal Dynamics
- **Weekday vs. Weekend**: Academic block load drops by **78%** on weekends, whereas hostel load increases by **28%**.
- **Exam Week Effect**: Central Library electricity consumption increases by **65%** during semester final exam weeks due to 24/7 occupancy.

---

### Slide 11 — EDA: Multi-Building Comparative Baseline

| Building Name | Category | Floor Area ($m^2$) | Avg Load ($kWh$) | Peak Load ($kWh$) | Annual EUI ($kWh/m^2/yr$) |
|---|---|---|---|---|---|
| **Academic Block 1** | Academic | 25,000 | 72.4 | 224.8 | 25.4 |
| **Hostel Boys Block** | Residential | 22,000 | 58.6 | 178.2 | 23.3 |
| **Hostel Girls Block** | Residential | 18,000 | 47.9 | 148.6 | 23.3 |
| **Central Library** | Academic/Study| 12,000 | 64.2 | 192.4 | 46.9 |
| **Dining Hall Mess** | Commercial | 8,000 | 50.4 | 215.1 | 55.2 |

#### Key Observation:
Dining Hall Mess and Central Library exhibit the highest Energy Use Intensity ($EUI$) per square meter owing to intensive kitchen equipment and continuous HVAC operation.

---

### Slide 12 — Data Quality & Integrity Assurance
#### Rigorous Pre-Flight Quality Verifications
1. **Timestamp Monotonicity**: Verified that all time intervals are strictly equidistant ($\Delta t = 1.0\text{ hr}$) with no missing time gaps.
2. **Range & Boundary Checks**: Confirmed no non-physical readings (e.g. negative energy draw, temperature $< -10^\circ\text{C}$ or $> 55^\circ\text{C}$).
3. **Outlier Treatment Philosophy**: Outliers were **not blindly deleted** from the dataset because uncharacteristically high readings during off-hours represent real energy anomalies that our ML anomaly engine must identify.

---

### Slide 13 — Data Preprocessing & Leakage-Free Splitting
#### Chronological Train / Validation / Test Splitting
To simulate real-world deployment and strictly eliminate lookahead data leakage, data was split chronologically:

```
├─── TRAINING SET: 70% (61,400 hours) ────┼── VAL: 15% (13,160) ──┼── TEST: 15% (13,160) ──┤
[ 2024-01-01 to 2025-05-26 ]               [ 2025-05-27 to 09-12 ]  [ 2025-09-13 to 12-31 ]
```
- **Training Set (70%)**: Used exclusively to learn regression weights and fit transformers.
- **Validation Set (15%)**: Used for hyperparameter tuning and cross-validation model selection.
- **Test Set (15%)**: Completely unseen future holdout data for final performance benchmarking.

---

### Slide 14 — Feature Engineering: 43 Predictive Features

```
┌───────────────────────────────────────┬────────────────────────────────────────────────────────┐
│ Feature Group                         │ Engineered Predictors                                  │
├───────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 1. Cyclical Time Encodings            │ hour_sin, hour_cos, dow_sin, dow_cos,                  │
│                                       │ month_sin, month_cos, is_business_hour                 │
├───────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 2. Thermodynamic & Weather Indices    │ cooling_degree_hours (CDH), heating_degree_hours (HDH),│
│                                       │ heat_index_c, solar_thermal_index, apparent_temp_c     │
├───────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 3. Autoregressive Lags                │ power_lag_1h, power_lag_2h, power_lag_24h (yesterday), │
│                                       │ power_lag_48h, power_lag_168h (last week), temp_lags   │
├───────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 4. Backward Rolling Statistics        │ power_rolling_mean_6h, std_6h, mean_24h, std_24h,      │
│                                       │ max_24h, min_24h, mean_168h, std_168h                  │
├───────────────────────────────────────┼────────────────────────────────────────────────────────┤
│ 5. Occupancy Interaction Terms        │ occ_x_temp, occ_x_exam, occ_x_business, area_sqm       │
└───────────────────────────────────────┴────────────────────────────────────────────────────────┘
```

---

### Slide 15 — ML Problem 1: Supervised Demand Forecasting
- **Supervised Task**: Multi-variate autoregressive time-series regression.
- **Inputs**: 43 engineered features capturing historical consumption inertia, thermal cooling strain, occupancy schedules, and calendar flags.
- **Target Output**: Precise continuous electricity consumption ($\hat{y}_{t}$ in $kWh$).
- **Success Criteria**: Test $R^2 > 0.85$, MAPE $< 10\%$, and Directional Trend Accuracy $> 90\%$.

---

### Slide 16 — Supervised Model Zoo Development
We benchmarked 6 diverse algorithms representing linear, bagging, boosting, and stacking paradigms:
1. **Persistence Baseline**: Naïve lag-24 baseline ($y_t = y_{t-24}$).
2. **Ridge Regression**: L2-penalized linear regression ($\alpha = 10.0$) with `StandardScaler`.
3. **Random Forest Regressor**: 100 bagged decision trees with feature subsampling and depth control.
4. **LightGBM Regressor**: Fast, histogram-binned gradient boosted decision trees with leaf-wise expansion.
5. **XGBoost Regressor**: Extreme Gradient Boosting with depth-wise tree regularization ($\eta = 0.05$).
6. **Stacking Ensemble**: Multi-layer stacking combiner using a Ridge meta-regressor blending Level-0 predictions.

---

### Slide 17 — Forecasting Benchmark Leaderboard (Test Set Results)

| Rank | Model Algorithm | Train $R^2$ | Val $R^2$ | Test $R^2$ | Test RMSE ($kWh$) | Test MAE ($kWh$) | Test MAPE ($\%$) | Directional Acc ($\%$) | Train Time |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 | **XGBoost Regressor** | **0.9330** | **0.9167** | **0.8855** | **15.133** | **5.587** | **8.35%** | **95.10%** | **0.71 s** |
| 🥈 | **LightGBM Regressor** | 0.9238 | 0.9166 | 0.8850 | 15.168 | 5.690 | 8.71% | 95.12% | 0.51 s |
| 🥉 | **Stacking Ensemble** | 0.9110 | 0.9130 | 0.8796 | 15.518 | 6.247 | 9.32% | 94.82% | 43.33 s |
| 4 | **Random Forest** | 0.9623 | 0.9079 | 0.8761 | 15.745 | 6.693 | 10.18% | 94.66% | 6.65 s |
| 5 | **Ridge Regression** | 0.8723 | 0.8738 | 0.8333 | 18.261 | 10.187 | 17.73% | 92.65% | 0.10 s |
| 6 | **Persistence Baseline** | 0.7014 | 0.6988 | 0.6252 | 27.384 | 13.766 | 18.71% | 91.02% | 0.00 s |

#### Core Takeaway:
**XGBoost Regressor** achieved the highest generalization score ($R^2 = 0.8855$, lowest RMSE of $15.133\text{ kWh}$, and $95.10\%$ directional accuracy) while executing training in under 1 second.

---

### Slide 18 — Hyperparameter Optimization & Cross-Validation
- **Cross-Validation Scheme**: 5-Fold `TimeSeriesSplit` respecting temporal causality.
- **Tuned Hyperparameters (XGBoost)**:
  - `n_estimators`: $200$
  - `learning_rate` ($\eta$): $0.05$ (lower shrinkage prevents overfitting)
  - `max_depth`: $6$ (balances non-linear feature interaction with tree complexity)
  - `subsample`: $0.85$ (stochastic row bagging)
  - `colsample_bytree`: $0.85$ (feature subsampling)
  - `tree_method`: `hist` (fast histogram-based split finding)

---

### Slide 19 — ML Problem 2: Unsupervised Anomaly Detection
#### Defining Campus Energy Anomalies
An anomaly is an operational observation that deviates significantly from the expected physical behavior conditioned on weather, occupancy, and historical patterns:
- **Off-Hours HVAC Leakage**: Chillers/heaters running at full load at 02:00 AM on Sunday in an empty academic hall.
- **Heavy Equipment Overload**: Sudden localized surges caused by faulty lab equipment, motor short-circuits, or unauthorized appliances.
- **Telemetry / Phase Drop**: Sudden uncharacteristic drop in power during busy classroom hours.
- **Thermal Inefficiency**: Chiller degradation requiring excessive power draw during high ambient heat.

---

### Slide 20 — Isolation Forest & Hybrid Residual Detection Architecture
#### Why Isolation Forest?
Isolation Forest isolates anomalies by randomly partitioning feature space. Since anomalies are few and structurally distinct, they require far fewer recursive tree splits to isolate than normal instances ($h(x) \ll c(n)$).

#### Dual-Engine Hybrid Formulation
$$\text{Anomaly Trigger} = (\text{Isolation Forest Score} < -0.05) \lor \left(\left|\frac{y - \hat{y}}{\sigma_{\text{res}}}\right| \ge 2.5\right)$$
- Combines multidimensional structural isolation with statistical studentized regression residuals.

---

### Slide 21 — Detected Energy Anomalies & Evaluation Metrics

```
  Electricity (kWh)
      ▲
  250 │                                  ● [CRITICAL: Equipment Short]
      │                                 /
  200 │              ● [HIGH: Off-Hours HVAC Leak]
      │             /
  150 │  ══════════════════════════════════════════════════ Expected Baseline (ŷ)
      │
  100 │
      │                                     ▼ [MEDIUM: Sensor Telemetry Drop]
   50 └────────────────────────────────────────────────────────► Time (Days)
```

#### Anomaly Engine Performance (Ground Truth Test Set):
- **Precision**: **99.64%** *(Virtually zero false alarms)*
- **Recall**: **72.02%** *(Successfully captures over 7 out of 10 complex anomalies)*
- **F1-Score**: **0.8361**
- **Classification Accuracy**: **98.81%**

---

### Slide 22 — Peak Consumption & Load Factor Analysis
- **Campus Peak Load Window**: Consistently occurs between **12:00 and 15:30** on weekdays during summer months (May–June) when ambient temperature exceeds $38^\circ\text{C}$ and class occupancy is at maximum.
- **Peak-to-Average Ratio (PAR)**:
  - Dining Hall Mess: $\text{PAR} = 4.27$ (High volatility meal peaks)
  - Academic Block 1: $\text{PAR} = 3.10$
  - Central Library: $\text{PAR} = 2.99$ (Stable continuous load profile)

---

### Slide 23 — Actionable Energy Risk Scoring (0–100 Index)

```
┌─────────────────┬───────────────┬────────────────────────────────────────────────────────┐
│ Risk Score Range│ Severity Tier │ Operational Protocol & Dispatch Action                 │
├─────────────────┼───────────────┼────────────────────────────────────────────────────────┤
│ 0 – 30          │ Low / Normal  │ Normal operations. Standard telemetry logging.         │
├─────────────────┼───────────────┼────────────────────────────────────────────────────────┤
│ 31 – 60         │ Medium        │ Automated notice to building supervisor. Re-check CT.  │
├─────────────────┼───────────────┼────────────────────────────────────────────────────────┤
│ 61 – 80         │ High          │ Immediate inspection of HVAC schedule relays & lights. │
├─────────────────┼───────────────┼────────────────────────────────────────────────────────┤
│ 81 – 100        │ Critical      │ Urgent technician dispatch: inspect breakers/chillers. │
└─────────────────┴───────────────┴────────────────────────────────────────────────────────┘
```

#### Financial & Carbon Impact Formulation:
$$\text{Wasted Energy} = \max(0, y_{\text{actual}} - \hat{y}_{\text{expected}})\text{ kWh}$$
$$\text{Financial Loss} = \text{Wasted Energy} \times ₹ 8.50/\text{kWh}$$
$$\text{Avoidable Carbon} = \text{Wasted Energy} \times 0.82\text{ kg CO}_2/\text{kWh}$$

---

### Slide 24 — Energy Intelligence Dashboard Architecture
The deployed **Streamlit Application** (`app.py`) provides 7 real-time operational tabs:
1. **🏢 Executive Command Center**: Campus-wide aggregate power gauge, live active alerts, and carbon tracker.
2. **📊 Exploratory Data Deep-Dive**: Diurnal profiles, day-of-week boxplots, and correlation matrices.
3. **🔮 Forecasting Studio**: Live multi-model comparison, dynamic horizon forecasting, and error analysis.
4. **⚠️ Anomaly Alert Center**: Color-coded timeline markers, automated root-cause explanations, and CSV export.
5. **🧠 Explainable AI (SHAP)**: Global feature importance and single-instance waterfall decomposition.
6. **🎛️ What-If Simulator**: Interactive heatwave, exam surge, and HVAC efficiency sliders.
7. **🏛️ Building Benchmarks**: Energy Use Intensity ($kWh/m^2/year$) and sustainability ratings.

---

### Slide 25 — Explainable AI (SHAP Global Feature Importance)
Using Tree SHAP, we decompose the model's predictive decision surface:

```
  Feature Name              Mean |SHAP Value| (kWh)
  ────────────────────────────────────────────────────────
  power_lag_24h             ████████████████████  12.96 kWh
  area_sqm                  ████████████          7.81 kWh
  occ_x_temp                ███████████           7.23 kWh
  power_lag_168h            ████████              5.07 kWh
  power_lag_1h              ████████              4.96 kWh
  cooling_degree_hours      ██████                3.84 kWh
  occupancy_rate            █████                 3.42 kWh
  hour_sin                  ████                  2.91 kWh
```

#### Interpretation:
- `power_lag_24h` acts as the primary temporal anchor.
- `area_sqm` sets the base thermodynamic scale.
- `occ_x_temp` captures the non-linear surge of occupants in hot ambient conditions requiring chilled air.

---

### Slide 26 — Explainable AI: Local Instance Waterfall Decomposition
For any specific individual hour (e.g. 2025-10-14 14:00 in Academic Block 1):
$$\hat{y} = \text{Base Value } (58.70\text{ kWh}) + \Delta_{\text{lag24h}} (+34.2\text{ kWh}) + \Delta_{\text{occ}\times\text{temp}} (+18.5\text{ kWh}) - \Delta_{\text{weekend}} (0.0) = \mathbf{111.4\text{ kWh}}$$

- **Actionable Explainability**: Rather than outputting a black-box number, the platform proves to facility engineers *why* a particular load is expected.

---

### Slide 27 — Error Analysis & Residual Diagnostics
1. **Error Normality**: Residual distribution ($e = y - \hat{y}$) is symmetric and Gaussian-like with zero mean ($\mu = 0.12\text{ kWh}, \sigma = 15.1\text{ kWh}$).
2. **Error Breakdown by Building**:
   - Central Library: Lowest relative error ($\text{MAPE} = 6.8\%$) due to stable schedules.
   - Dining Hall Mess: Highest relative error ($\text{MAPE} = 11.2\%$) during rapid meal transitions.
3. **Failure Modes**: Minor heteroskedasticity during abrupt weather shifts (e.g. sudden monsoon cloudbursts causing temperature to plunge $10^\circ\text{C}$ in 30 minutes).

---

### Slide 28 — End-to-End Reproducible ML Pipeline

```
[ Raw CSV Data ] ──► [ Pipeline.py ] ──► [ Chronological Split ] ──► [ Model Training Zoo ]
                           │                                                 │
                           ▼                                                 ▼
                 [ Feature Transforms ]                            [ Serialized Artifacts ]
                 (Lags, CDH, Cyclic)                               (.joblib Models & Encoders)
                           │                                                 │
                           └──────────────────► [ Streamlit App ] ◄──────────┘
```

- Fully serialized pipeline saving models to `saved_models/` for 1-click execution via `python train.py`.

---

### Slide 29 — Technology Stack Justification

```
┌──────────────────────┬────────────────────────┬────────────────────────────────────────┐
│ Module Layer         │ Technology / Library   │ Technical Justification                │
├──────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ Core Data Processing │ Python 3.11, Pandas,   │ High-speed vectorized data transforms  │
│                      │ NumPy, Scipy           │ and time-series rolling computations.  │
├──────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ Machine Learning     │ Scikit-Learn, XGBoost, │ State-of-the-art gradient boosted tree │
│                      │ LightGBM, Joblib       │ ensembles with second-order Taylor exp.│
├──────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ Anomaly & XAI        │ Isolation Forest, SHAP │ Unsupervised subspace isolation and    │
│                      │ TreeExplainer          │ Shapley additive game-theoretic XAI.   │
├──────────────────────┼────────────────────────┼────────────────────────────────────────┤
│ UI & Deployment      │ Streamlit, Plotly,     │ High-performance reactive dashboard    │
│                      │ Git & GitHub           │ with interactive WebGL visualizations. │
└──────────────────────┴────────────────────────┴────────────────────────────────────────┘
```

---

### Slide 30 — Deployment & Live Interactive Verification
- **Web Interface**: Deployed on `http://localhost:8501`.
- **Latency**: Sub-second prediction latency ($< 35\text{ ms}$ per building forecast).
- **Interactive Capabilities**: Users can dynamically filter buildings, run real-time what-if scenario simulations, inspect SHAP waterfalls, and export detected anomaly logs.

---

### Slide 31 — GitHub Repository Organization

```
campus-energy-intelligence-and-anomaly-detection/
├── app.py                                  # Streamlit Interactive Web Application
├── train.py                                # Automated CLI Pipeline Runner
├── requirements.txt                        # Python dependencies
├── README.md                               # Comprehensive Project Report
├── VIVA_PREPARATION_GUIDE.md               # 3-Student Viva Defence Guide
├── PRESENTATION_SLIDES_COMPLETE_40.md      # Full 40-Slide Presentation Deck
├── data/                                   # Processed datasets and metrics JSON
├── notebooks/                              # Jupyter Notebook with full step-by-step EDA
├── saved_models/                           # Serialized XGBoost & Isolation Forest models
└── src/                                    # Modular Source Code Package
```
**Live GitHub Repo**: `https://github.com/Amit0730/campus-energy-intelligence-and-anomaly-detection`

---

### Slide 32 — Summary of Experimental Results

```
┌──────────────────────────────────────────────┬────────────────────────────────────────┐
│ Metric Description                           │ Empirical Experimental Value           │
├──────────────────────────────────────────────┼────────────────────────────────────────┤
│ Total Monitored Operational Dataset          │ 87,720 hourly records (2 Full Years)   │
│ Total Features Engineered                    │ 43 domain-specific predictors          │
│ Best Supervised Forecasting Model            │ XGBoost Regressor                      │
│ Test Set Coefficient of Determination (R²)   │ 0.8855 (Val: 0.9167, Train: 0.9330)    │
│ Test Set Mean Absolute Error (MAE)           │ 5.587 kWh                              │
│ Directional Trend Accuracy                   │ 95.10%                                 │
│ Anomaly Detection Precision                  │ 99.64%                                 │
│ Anomaly Detection F1-Score                   │ 0.8361                                 │
│ Total Identified Energy Wastage (Test Set)   │ 14,820 kWh (₹ 1,25,970 avoidable cost) │
└──────────────────────────────────────────────┴────────────────────────────────────────┘
```

---

### Slide 33 — Key Empirical Insights
1. **Lag-24 & 168 Anchoring**: Campus energy follows strong daily and weekly circadian rhythms; yesterday's power is the single strongest predictor.
2. **Thermal Load Sensitivity**: Above $22^\circ\text{C}$, every $1.0^\circ\text{C}$ temperature increase triggers a non-linear $4.5\% - 7.2\%$ surge in campus chiller load.
3. **Off-Hours Leakage Dominance**: **64%** of detected financial wastage occurred between 23:00 and 05:00 due to unattended HVAC systems in empty buildings.

---

### Slide 34 — Project Limitations & Assumptions
1. **Synthetic Sensor Augmentation**: Real-world university sub-metering data was simulated using physics-based empirical thermodynamics due to proprietary campus BMS telemetry access limits.
2. **Point Forecasting vs. Probabilistic Quantiles**: Models generate point estimates; future iterations will incorporate quantile regression for prediction intervals.
3. **Static Tariffs**: Assumes a flat commercial tariff of ₹8.5/kWh rather than dynamic time-of-day (ToD) peak-pricing tariffs.

---

### Slide 35 — Future Scope & Extensions
1. **IoT Smart Meter Integration**: Direct MQTT/Modbus ingestion from campus digital smart meters.
2. **Automated Relay Control**: Closed-loop Building Automation System (BAS) integration to automatically cut power to idle zones.
3. **Renewable Solar Microgrid Optimization**: Forecasting campus rooftop solar output and scheduling heavy lab loads during peak solar generation.
4. **Campus Digital Twin**: 3D interactive building twin with live heatmaps.

---

### Slide 36 — Conclusion
- Successfully developed and deployed an end-to-end Machine Learning intelligence platform for university campus electricity management.
- Delivered state-of-the-art forecasting ($R^2 = 0.8855$, $95.10\%$ trend accuracy) and high-precision anomaly detection ($99.64\%$ precision).
- Transformed raw time-series data into actionable financial, operational, and sustainability insights through an intuitive, explainable dashboard.

---

### Slide 37 — Team Member Technical Contributions

```
┌──────────────────────┬────────────────────────────────────────────────────────────────┐
│ Team Member          │ Technical Modules Owned & Defended                             │
├──────────────────────┼────────────────────────────────────────────────────────────────┤
│ Student 1            │ Problem Formulation, Synthetic Data Generator Engine, EDA,     │
│ (Data & Preprocessing)│ 43-Feature Engineering Pipeline & Leakage-Free Temporal Splits.│
├──────────────────────┼────────────────────────────────────────────────────────────────┤
│ Student 2            │ Supervised Model Zoo Development (6 Algorithms), TimeSeriesCV, │
│ (ML & Optimization)  │ Hyperparameter Optimization & Residual Diagnostic Analysis.    │
├──────────────────────┼────────────────────────────────────────────────────────────────┤
│ Student 3            │ Hybrid Isolation Forest Anomaly Engine, SHAP XAI Visualizers,  │
│ (Deployment & XAI)   │ Streamlit Dashboard, What-If Simulator & GitHub Documentation. │
└──────────────────────┴────────────────────────────────────────────────────────────────┘
```

---

### Slide 38 — Academic & Technical References
1. *Scikit-learn: Machine Learning in Python*, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
2. *XGBoost: A Scalable Tree Boosting System*, Chen & Guestrin, KDD '16, 2016.
3. *A Unified Approach to Interpreting Model Predictions (SHAP)*, Lundberg & Lee, NeurIPS 2017.
4. *Isolation Forest*, Liu, Ting & Zhou, IEEE ICDM, 2008.
5. *ASHRAE Guideline 14: Measurement of Energy, Demand, and Water Savings*, 2014.

---

### Slide 39 — Verification & Demonstration Proof
- **Live Local Web App**: `http://localhost:8501`
- **GitHub Repository**: `https://github.com/Amit0730/campus-energy-intelligence-and-anomaly-detection`
- **Jupyter Notebook**: `notebooks/Campus_Energy_Intelligence_EDA_Modeling.ipynb`
- **Pipeline Runner**: `python train.py`

---

### Slide 40 — Thank You & Viva Q&A
# Thank You!
### Open for Questions from the Evaluation Panel
*All code, models, and documentation are available in the public repository.*
