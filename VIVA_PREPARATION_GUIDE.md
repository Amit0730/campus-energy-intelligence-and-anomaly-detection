# 🎓 INT395 Viva & Presentation Preparation Guide
## Project 15: Campus Energy Consumption Intelligence & Anomaly Detection
**Total Marks: 35 Marks | Aligned with Official Rubrics**

---

### Rubric Component 1: Problem Understanding & Formulation (3 Marks)
**Expected Technical Defence**:
- **What problem does this project solve?**
  University campuses are multi-zone microgrids containing distinct building types (academic lecture halls, computer labs, 24/7 libraries, student residential dorms, and high-load commercial dining halls). Electricity load is highly non-linear, driven by weather extremes (temperature, solar radiation), occupancy shifts (class schedules, exam weeks, vacations), and equipment usage. Without intelligence, off-hours HVAC runaways, faulty relays, and equipment short-circuits cause massive energy wastage, expensive peak-demand utility penalties, and unnecessary carbon emissions.
- **What is the ML formulation?**
  A dual-engine architecture:
  1. **Supervised Regression**: Hourly time-series electricity forecasting ($\hat{y}_{t}$ in $kWh$) for each building and the aggregate campus.
  2. **Unsupervised & Residual Anomaly Detection**: Hybrid isolation forest and statistical studentized residual detector ($z = \frac{y - \hat{y}}{\sigma}$) to flag anomalous consumption, assign severity tiers (Low, Medium, High, Critical), calculate financial wastage ($₹$), and diagnose root causes.

---

### Rubric Component 2: Dataset, EDA & Data Quality Analysis (4 Marks)
**Expected Technical Defence**:
- **Dataset characteristics**:
  - 2 full years of continuous hourly records (87,600 building-hours across 5 buildings).
  - Features: ambient temperature ($^\circ\text{C}$), relative humidity ($\%$), solar radiation ($W/m^2$), wind speed ($m/s$), occupancy rate ($0.0-1.0$), academic flags (`is_exam_week`, `is_vacation`, `is_holiday`, `is_weekend`), sub-metering channels (HVAC, lighting, equipment), and realistic anomaly injections.
- **Key EDA Findings**:
  - Academic buildings exhibit heavy diurnal demand strictly between 08:00 and 18:00 on weekdays, dropping by ~80% on weekends.
  - Hostels show an inverse profile (peaks during 06:30–09:00 and 18:00–00:00, remaining high throughout weekends).
  - Temperature shows a strong non-linear polynomial relationship with energy due to HVAC cooling loads above the 22°C baseline.

---

### Rubric Component 3: Data Preprocessing & Feature Engineering (5 Marks)
**Expected Technical Defence**:
- **How was data leakage prevented?**
  Chronological splitting was strictly enforced (first 70% train, next 15% validation, final 15% completely unseen future test set). No future information was used during feature normalization or rolling window calculations.
- **Engineered Features**:
  1. *Cyclic trigonometric transforms*: $\sin/\cos$ mappings for hour (period 24), day of week (period 7), and month (period 12) so the model recognizes 23:00 is adjacent to 00:00.
  2. *Thermodynamic indicators*: Cooling Degree Hours ($\text{CDH} = \max(0, T - 22)$), Heating Degree Hours ($\text{HDH} = \max(0, 16 - T)$), and Rothfusz Heat Index.
  3. *Autoregressive Lags*: $t-1\text{h}, t-2\text{h}, t-24\text{h}$ (yesterday), $t-168\text{h}$ (same hour last week) computed per building.
  4. *Rolling statistics*: Backward rolling mean, standard deviation, min, and max over 6h, 24h, and 168h windows.

---

### Rubric Component 4: Model Development & Comparative Analysis (5 Marks)
**Expected Technical Defence**:
- **Why compare multiple models?**
  To establish an empirical baseline and justify why non-linear tree-based ensembles and stacking architectures outperform linear and persistence heuristics.
- **Benchmarked Models**:
  1. *Persistence Baseline*: Predicts same hour yesterday ($t-24\text{h}$).
  2. *Ridge Regression*: Linear model with L2 regularization to prevent multicollinearity among lag features.
  3. *Random Forest Regressor*: Bagging ensemble capturing complex non-linear feature interactions without overfitting.
  4. *LightGBM Regressor*: Fast, leaf-wise gradient boosting using histogram binning for superior speed and memory efficiency.
  5. *XGBoost Regressor*: Depth-wise extreme gradient boosting with second-order Taylor expansion loss approximation.
  6. *Stacking Ensemble*: Meta-regressor blending predictions from Ridge, RF, LightGBM, and XGBoost.

---

### Rubric Component 5: Ensemble Learning & Hyperparameter Optimization (4 Marks)
**Expected Technical Defence**:
- **Ensemble architecture**:
  Stacking Regressor using a multi-level learning framework. Level-0 base estimators (Ridge, Random Forest, LightGBM, XGBoost) generate out-of-fold predictions that feed into a Level-1 Ridge meta-learner, minimizing generalization variance.
- **Optimization**:
  Cross-validation performed with `TimeSeriesSplit` (respecting temporal causality) optimizing learning rate ($\eta = 0.05$), tree depth, subsample ratio ($0.85$), and colsample_bytree.

---

### Rubric Component 6: Model Evaluation, Validation & Error Analysis (3 Marks)
**Expected Technical Defence**:
- **Evaluation Metrics Used**:
  - $R^2$ Score (Variance explained: $> 0.90$)
  - RMSE & MAE (Absolute error magnitude in $kWh$)
  - MAPE ($\%$ relative error across load scales)
  - Directional Accuracy ($\%$ correct hourly trend direction)
- **Residual Analysis**:
  Residual errors ($e_i = y_i - \hat{y}_i$) follow an approximately Gaussian distribution centered at zero. Error heteroskedasticity occurs during extreme transition seasons (e.g. sudden monsoon thunderstorms causing rapid temperature drops).

---

### Rubric Component 7: Explainable AI / Model Interpretability (3 Marks)
**Expected Technical Defence**:
- **Why SHAP (SHapley Additive exPlanations)?**
  Tree SHAP provides mathematically proven game-theoretic properties (efficiency, symmetry, additivity) satisfying:
  $$\hat{y} = \phi_0 + \sum_{j=1}^{M} \phi_j$$
  where $\phi_0$ is the base expected campus load and $\phi_j$ is the exact impact of feature $j$.
- **Key Interpretability Insights**:
  - `power_lag_24h` and `power_rolling_mean_24h` form the historical load anchor.
  - `temperature_c` and `cooling_degree_hours` drive daytime peak HVAC power spikes.
  - `hour_sin` and `occupancy_rate` govern diurnal shifts between active class hours and night standby.

---

### Rubric Component 8: End-to-End Pipeline & Technical Integration (3 Marks)
**Expected Technical Defence**:
- Complete separation of concerns:
  - `src/data_generator.py` $\rightarrow$ Synthetic physics-based data engine
  - `src/preprocessing.py` $\rightarrow$ Scikit-learn feature pipeline
  - `src/models.py` $\rightarrow$ Model zoo & metric evaluation
  - `src/anomaly_detector.py` $\rightarrow$ Isolation Forest & residual diagnostic engine
  - `src/explainability.py` $\rightarrow$ SHAP TreeExplainer wrapper
  - `src/pipeline.py` $\rightarrow$ Full lifecycle orchestrator & joblib serializer
  - `train.py` $\rightarrow$ CLI entrypoint

---

### Rubric Component 9: Deployment & Working Prototype (2 Marks)
**Expected Technical Defence**:
- Deployed as a full-featured **Streamlit Web Application** (`app.py`) featuring 7 interactive operational modules:
  1. *Executive Command Center*: Live load gauges, building status, and avoidable wastage KPI cards.
  2. *Exploratory Data Analysis*: Interactive Plotly diurnal curves, correlation matrices, and sub-metering stack plots.
  3. *Forecasting Studio*: Multi-model comparison, dynamic horizon forecasting, and error residual inspector.
  4. *Anomaly Alert Center*: Color-coded timeline markers, automated root-cause diagnostics, and financial loss calculations.
  5. *Explainable AI Studio*: Global feature importance and single-instance waterfall decomposition.
  6. *What-If Simulator*: Interactive heatwave, exam week, and efficiency scenario projections.
  7. *Building Benchmarking*: Energy Use Intensity ($kWh/m^2/year$) and sustainability ratings.

---

### Rubric Component 10: GitHub, Documentation & Viva (3 Marks)
**Expected Technical Defence**:
- Complete clean repository structure with `README.md`, `requirements.txt`, modular `src/` code, standalone Jupyter Notebook (`notebooks/Campus_Energy_Intelligence_EDA_Modeling.ipynb`), and serialized artifacts in `saved_models/`.
