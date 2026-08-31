"""
Campus Energy Consumption Intelligence & Anomaly Detection
Streamlit Interactive Dashboard Deployment (Project 15)
INT395: Supervised Learning Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# Ensure root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import load_pipeline_artifacts
from src.explainability import EnergyExplainer

# Streamlit Page Config
st.set_page_config(
    page_title="Campus Energy Intelligence & Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 18px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    #MainMenu {visibility: hidden; display: none !important;}
    footer {visibility: hidden; display: none !important;}
    header {visibility: hidden; display: none !important;}
    [data-testid="stToolbar"] {visibility: hidden; display: none !important;}
    [data-testid="stDecoration"] {visibility: hidden; display: none !important;}
    [data-testid="stHeader"] {visibility: hidden; display: none !important;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_artifacts():
    """Loads and caches all pipeline artifacts."""
    return load_pipeline_artifacts(data_dir="data", model_dir="saved_models")


def main():
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/electricity.png", width=70)
    st.sidebar.title("Campus Energy AI")
    st.sidebar.divider()

    menu = st.sidebar.radio(
        "Navigation Modules",
        [
            "🏢 Executive Overview",
            "📊 Exploratory Data Analysis",
            "🔮 Demand Forecasting & Models",
            "⚠️ Anomaly Detection & Wastage",
            "🧠 Explainable AI (SHAP)",
            "🎛️ What-If Scenario Simulator",
            "🏛️ Building Benchmarks & EUI"
        ]
    )

    try:
        artifacts = get_artifacts()
    except Exception as e:
        st.error(f"⚠️ Artifacts not found or pipeline not yet executed. Please run `python train.py` first.\n\nError: {e}")
        return

    featured_df = artifacts["featured_df"]
    test_df = artifacts["test_results_df"]
    models = artifacts["models"]
    best_model_name = artifacts["best_model_name"]
    metrics_summary = artifacts["metrics_summary"]
    building_profiles = artifacts["building_profiles"]
    shap_importance_df = artifacts["shap_importance_df"]

    # Header
    st.markdown('<div class="main-header">⚡ Campus Energy Consumption Intelligence & Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-driven predictive electricity load forecasting, real-time wastage anomaly detection, and explainable insights across campus facilities.</div>', unsafe_allow_html=True)

    # -------------------------------------------------------------
    # TAB 1: EXECUTIVE OVERVIEW
    # -------------------------------------------------------------
    if menu == "🏢 Executive Overview":
        st.subheader("🏢 Campus Energy Operational Command Center")
        
        total_kwh_tested = test_df["total_power_kwh"].sum()
        total_anomalies = (test_df["detected_anomaly"] == 1).sum()
        total_wasted_kwh = test_df["wasted_energy_kwh"].sum()
        total_wasted_cost = test_df["financial_loss_inr"].sum()
        total_excess_co2 = test_df["excess_co2_kg"].sum()

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Monitored Hours", f"{len(test_df):,} hrs", "5 Campus Buildings")
        col2.metric("Top Model Test R²", f"{metrics_summary['model_comparison'][0]['Test R2']:.4f}", f"{best_model_name}")
        col3.metric("Detected Anomalies", f"{total_anomalies:,} events", f"{(total_anomalies/len(test_df)*100):.1f}% anomalous")
        col4.metric("Energy Wastage Loss", f"₹ {total_wasted_cost:,.0f}", f"{total_wasted_kwh:,.0f} kWh wasted")
        col5.metric("Excess CO₂ Emissions", f"{total_excess_co2/1000:,.1f} tons", "Avoidable footprint")

        st.markdown("---")
        
        c_left, c_right = st.columns([2, 1])
        
        with c_left:
            st.markdown("#### 📈 Campus Aggregated Hourly Load Profile (Recent 14 Days)")
            campus_hourly = test_df.groupby("timestamp")[["total_power_kwh", "forecast_kwh"]].sum().reset_index().tail(336)
            
            fig_agg = go.Figure()
            fig_agg.add_trace(go.Scatter(
                x=campus_hourly["timestamp"], y=campus_hourly["total_power_kwh"],
                mode="lines", name="Actual Power (kWh)", line=dict(color="#2563EB", width=2)
            ))
            fig_agg.add_trace(go.Scatter(
                x=campus_hourly["timestamp"], y=campus_hourly["forecast_kwh"],
                mode="lines", name="AI Baseline Forecast (kWh)", line=dict(color="#10B981", width=2, dash="dot")
            ))
            fig_agg.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                xaxis_title="Timeline", yaxis_title="Total Campus Load (kWh)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_agg, use_container_width=True)

        with c_right:
            st.markdown("#### 🏛️ Monitored Facilities Overview")
            b_summary = []
            for b_name, b_info in building_profiles.items():
                b_records = test_df[test_df["building_id"] == b_name]
                avg_load = b_records["total_power_kwh"].mean()
                peak_load = b_records["total_power_kwh"].max()
                anoms = (b_records["detected_anomaly"] == 1).sum()
                b_summary.append({
                    "Building": b_name.replace("_", " "),
                    "Type": b_info["type"],
                    "Area (m²)": f"{b_info['area_sqm']:,}",
                    "Avg Load": f"{avg_load:.1f} kWh",
                    "Peak Load": f"{peak_load:.1f} kWh",
                    "Alerts": anoms
                })
            st.dataframe(pd.DataFrame(b_summary), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 2: EXPLORATORY DATA ANALYSIS
    # -------------------------------------------------------------
    elif menu == "📊 Exploratory Data Analysis":
        st.subheader("📊 Campus Energy Consumption & Weather EDA")
        
        b_select = st.selectbox("Select Campus Building", list(building_profiles.keys()), format_func=lambda x: x.replace("_", " "))
        b_df = featured_df[featured_df["building_id"] == b_select].copy()
        
        eda_tab1, eda_tab2, eda_tab3, eda_tab4 = st.tabs([
            "🕒 Diurnal & Weekly Patterns", "🌡️ Weather Correlations", "⚡ Sub-Metering Breakdown", "📅 Calendar Heatmap"
        ])
        
        with eda_tab1:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("##### 24-Hour Average Consumption Curve (Weekday vs Weekend)")
                diurnal = b_df.groupby(["hour", "is_weekend"])["total_power_kwh"].mean().reset_index()
                diurnal["Day Type"] = diurnal["is_weekend"].map({0: "Weekday", 1: "Weekend"})
                fig_diurn = px.line(
                    diurnal, x="hour", y="total_power_kwh", color="Day Type",
                    title="Diurnal Load Profile",
                    labels={"hour": "Hour of Day (0-23)", "total_power_kwh": "Mean Consumption (kWh)"},
                    color_discrete_map={"Weekday": "#3B82F6", "Weekend": "#F59E0B"}
                )
                st.plotly_chart(fig_diurn, use_container_width=True)
                
            with col_d2:
                st.markdown("##### Day of Week Load Distribution")
                dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                fig_box = px.box(
                    b_df, x="day_name", y="total_power_kwh", category_orders={"day_name": dow_order},
                    color="day_name", title="Load Spread by Day of Week",
                    labels={"day_name": "Day", "total_power_kwh": "Electricity (kWh)"}
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True)

        with eda_tab2:
            st.markdown("##### Outdoor Temperature & Solar Radiation vs Building Power")
            col_w1, col_w2 = st.columns(2)
            with col_w1:
                fig_scat = px.scatter(
                    b_df.sample(2000, random_state=42),
                    x="temperature_c", y="total_power_kwh", color="occupancy_rate",
                    color_continuous_scale="Viridis",
                    title="Electricity Consumption vs Outdoor Temperature (°C)",
                    labels={"temperature_c": "Temperature (°C)", "total_power_kwh": "Electricity (kWh)", "occupancy_rate": "Occupancy"}
                )
                st.plotly_chart(fig_scat, use_container_width=True)
            with col_w2:
                corr_cols = ["total_power_kwh", "temperature_c", "humidity_pct", "solar_radiation_wm2", "occupancy_rate", "cooling_degree_hours"]
                corr_mat = b_df[corr_cols].corr()
                fig_corr = px.imshow(
                    corr_mat, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                    title="Correlation Matrix with Environmental Features"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        with eda_tab3:
            st.markdown("##### Sub-Metering Component Breakdown (HVAC, Lighting, Equipment)")
            sub_components = b_df[["timestamp", "hvac_power_kwh", "lighting_power_kwh", "equipment_power_kwh"]].tail(168)
            fig_sub = go.Figure()
            fig_sub.add_trace(go.Scatter(x=sub_components["timestamp"], y=sub_components["hvac_power_kwh"], name="HVAC", stackgroup="one", fillcolor="rgba(239, 68, 68, 0.6)"))
            fig_sub.add_trace(go.Scatter(x=sub_components["timestamp"], y=sub_components["lighting_power_kwh"], name="Lighting", stackgroup="one", fillcolor="rgba(234, 179, 8, 0.6)"))
            fig_sub.add_trace(go.Scatter(x=sub_components["timestamp"], y=sub_components["equipment_power_kwh"], name="Equipment / Plug Load", stackgroup="one", fillcolor="rgba(59, 130, 246, 0.6)"))
            fig_sub.update_layout(title="Sub-System Stacked Energy Draw (Past 7 Days)", yaxis_title="Power (kWh)", xaxis_title="Timestamp")
            st.plotly_chart(fig_sub, use_container_width=True)

        with eda_tab4:
            st.markdown("##### Hourly Load Heatmap Matrix (Hour vs Day of Week)")
            pivot_table = b_df.pivot_table(index="day_name", columns="hour", values="total_power_kwh", aggfunc="mean").reindex(
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            )
            fig_heat = px.imshow(pivot_table, labels=dict(x="Hour of Day", y="Day of Week", color="Avg kWh"), color_continuous_scale="Turbo")
            st.plotly_chart(fig_heat, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 3: DEMAND FORECASTING & MODEL COMPARISON
    # -------------------------------------------------------------
    elif menu == "🔮 Demand Forecasting & Models":
        st.subheader("🔮 Supervised Energy Forecasting & Model Comparative Benchmark")
        
        st.markdown("#### 🏆 Model Benchmark Leaderboard (INT395 Evaluation Rubric)")
        comp_df = pd.DataFrame(metrics_summary["model_comparison"])
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        st.markdown("#### 🔬 Interactive Time-Series Forecasting & Error Inspection")
        
        c_b, c_m, c_days = st.columns([1.5, 1.5, 1])
        with c_b:
            b_choice = st.selectbox("Select Building", list(building_profiles.keys()), format_func=lambda x: x.replace("_", " "))
        with c_m:
            model_choice = st.selectbox("Select Trained Model", list(models.keys()), index=list(models.keys()).index(best_model_name) if best_model_name in models else 0)
        with c_days:
            forecast_window = st.slider("Test Horizon (Days)", min_value=3, max_value=30, value=7)
            
        b_test = test_df[test_df["building_id"] == b_choice].copy()
        
        # Make dynamic prediction with selected model
        selected_model = models[model_choice]
        X_test_b = b_test[artifacts["feature_cols"]]
        dyn_preds = selected_model.predict(X_test_b)
        b_test["chosen_forecast_kwh"] = np.round(dyn_preds, 2)
        
        # Display slice
        slice_b = b_test.tail(forecast_window * 24)
        
        fig_fore = go.Figure()
        fig_fore.add_trace(go.Scatter(
            x=slice_b["timestamp"], y=slice_b["total_power_kwh"],
            mode="lines", name="Ground Truth Actual (kWh)", line=dict(color="#1F2937", width=2)
        ))
        fig_fore.add_trace(go.Scatter(
            x=slice_b["timestamp"], y=slice_b["chosen_forecast_kwh"],
            mode="lines", name=f"{model_choice} Forecast", line=dict(color="#2563EB", width=2.5, dash="dot")
        ))
        fig_fore.update_layout(
            title=f"Actual vs Predicted Load ({b_choice.replace('_', ' ')}) - Last {forecast_window} Days",
            xaxis_title="Timestamp", yaxis_title="Electricity (kWh)",
            hovermode="x unified", height=400
        )
        st.plotly_chart(fig_fore, use_container_width=True)
        
        # Residual Error Analysis
        residuals = slice_b["total_power_kwh"] - slice_b["chosen_forecast_kwh"]
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_res_dist = px.histogram(
                residuals, nbins=35, title="Residual Error Distribution (e = y - ŷ)",
                labels={"value": "Error (kWh)"}, color_discrete_sequence=["#6366F1"]
            )
            st.plotly_chart(fig_res_dist, use_container_width=True)
        with col_r2:
            fig_qq = px.scatter(
                x=slice_b["chosen_forecast_kwh"], y=slice_b["total_power_kwh"],
                title="Predicted vs Actual Scatter Plot (Goodness-of-fit)",
                labels={"x": "Predicted (kWh)", "y": "Actual (kWh)"}
            )
            min_val = min(slice_b["chosen_forecast_kwh"].min(), slice_b["total_power_kwh"].min())
            max_val = max(slice_b["chosen_forecast_kwh"].max(), slice_b["total_power_kwh"].max())
            fig_qq.add_shape(type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val, line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_qq, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 4: ANOMALY DETECTION & WASTAGE ALERT CENTER
    # -------------------------------------------------------------
    elif menu == "⚠️ Anomaly Detection & Wastage":
        st.subheader("⚠️ Campus Energy Anomaly Detection & Wastage Diagnostic Center")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            b_filter = st.selectbox("Building Filter", ["All Buildings"] + list(building_profiles.keys()), format_func=lambda x: x.replace("_", " "))
        with col_f2:
            sev_filter = st.multiselect("Severity Filter", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium"])
        with col_f3:
            search_cause = st.text_input("Search Root Cause", "")

        anom_data = test_df.copy()
        if b_filter != "All Buildings":
            anom_data = anom_data[anom_data["building_id"] == b_filter]
        if sev_filter:
            anom_data = anom_data[anom_data["detected_severity"].isin(sev_filter)]
        if search_cause:
            anom_data = anom_data[anom_data["root_cause"].str.contains(search_cause, case=False, na=False)]
            
        anom_data = anom_data[anom_data["detected_anomaly"] == 1]
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Anomalies Matching Filter", f"{len(anom_data):,} events")
        col_m2.metric("Critical Alerts", f"{(anom_data['detected_severity']=='Critical').sum():,}")
        col_m3.metric("Estimated Wasted Energy", f"{anom_data['wasted_energy_kwh'].sum():,.1f} kWh")
        col_m4.metric("Financial Loss (₹)", f"₹ {anom_data['financial_loss_inr'].sum():,.0f}")
        
        st.markdown("---")
        st.markdown("#### 🚨 Anomaly Event Timeline & Diagnostic Annotations")
        
        sample_b_name = list(building_profiles.keys())[0] if b_filter == "All Buildings" else b_filter
        sample_anom_view = test_df[test_df["building_id"] == sample_b_name].tail(720) # 30 days
        
        fig_anom = go.Figure()
        fig_anom.add_trace(go.Scatter(
            x=sample_anom_view["timestamp"], y=sample_anom_view["total_power_kwh"],
            mode="lines", name="Actual Power", line=dict(color="#94A3B8", width=1.5)
        ))
        fig_anom.add_trace(go.Scatter(
            x=sample_anom_view["timestamp"], y=sample_anom_view["forecast_kwh"],
            mode="lines", name="Expected Baseline", line=dict(color="#10B981", width=1.5, dash="dot")
        ))
        
        anom_points = sample_anom_view[sample_anom_view["detected_anomaly"] == 1]
        sev_color_map = {"Critical": "#DC2626", "High": "#EA580C", "Medium": "#CA8A04", "Low": "#2563EB"}
        
        for sev, group in anom_points.groupby("detected_severity"):
            fig_anom.add_trace(go.Scatter(
                x=group["timestamp"], y=group["total_power_kwh"],
                mode="markers", name=f"{sev} Anomaly",
                marker=dict(size=9, color=sev_color_map.get(sev, "red"), symbol="circle-open-dot", line=dict(width=2))
            ))
            
        fig_anom.update_layout(
            title=f"Anomaly Events Detected on {sample_b_name.replace('_', ' ')} (Past 30 Days)",
            xaxis_title="Date", yaxis_title="Power (kWh)", height=400
        )
        st.plotly_chart(fig_anom, use_container_width=True)
        
        st.markdown("#### 📋 Actionable Anomaly Investigation Log")
        display_cols = ["timestamp", "building_id", "detected_severity", "total_power_kwh", "forecast_kwh", "wasted_energy_kwh", "financial_loss_inr", "root_cause", "recommended_action"]
        st.dataframe(anom_data[display_cols].sort_values("timestamp", ascending=False).head(50), use_container_width=True, hide_index=True)

    # -------------------------------------------------------------
    # TAB 5: EXPLAINABLE AI (SHAP)
    # -------------------------------------------------------------
    elif menu == "🧠 Explainable AI (SHAP)":
        st.subheader("🧠 Explainable AI (XAI) with SHAP (SHapley Additive exPlanations)")
        
        col_x1, col_x2 = st.columns([1, 1])
        with col_x1:
            st.markdown("#### 🌐 Global Feature Importance (Mean |SHAP| Value)")
            fig_shap_glob = px.bar(
                shap_importance_df.head(15),
                x="Mean_Abs_SHAP", y="Feature", orientation="h",
                color="Mean_Abs_SHAP", color_continuous_scale="Blues",
                title="Top 15 Influential Predictors Across Entire Campus"
            )
            fig_shap_glob.update_layout(yaxis=dict(autorange="reversed"), height=450)
            st.plotly_chart(fig_shap_glob, use_container_width=True)

        with col_x2:
            st.markdown("#### 🔬 Local Instance Explainer (Waterfall Decomposition)")
            b_exp = st.selectbox("Building", list(building_profiles.keys()), key="exp_b", format_func=lambda x: x.replace("_", " "))
            b_recs = test_df[test_df["building_id"] == b_exp].tail(100)
            selected_ts = st.selectbox("Timestamp Instance", b_recs["timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist())
            
            chosen_row = b_recs[b_recs["timestamp"].dt.strftime("%Y-%m-%d %H:%M") == selected_ts].iloc[0]
            
            tree_model_name = "XGBoost Regressor" if "XGBoost Regressor" in models else best_model_name
            tree_model = models.get(tree_model_name, models[best_model_name])
            explainer = EnergyExplainer(tree_model, artifacts["feature_cols"])
            explanation = explainer.explain_instance(chosen_row)
            
            base_v = explanation["base_value_kwh"]
            pred_v = chosen_row.get("forecast_kwh", chosen_row["total_power_kwh"])
            
            st.info(f"**Campus Mean Baseline Base Value:** {base_v:.2f} kWh  ➡️  **Final Prediction for Hour:** {pred_v:.2f} kWh")
            
            top_contribs = explanation["all_contributions"].head(8)
            fig_waterfall = go.Figure(go.Waterfall(
                name="SHAP Impact", orientation="v",
                measure=["relative"] * len(top_contribs),
                x=top_contribs["Feature"],
                y=top_contribs["SHAP_Impact_kWh"],
                textposition="outside",
                text=[f"{v:+.1f}" for v in top_contribs["SHAP_Impact_kWh"]],
                connector={"line": {"color": "rgb(63, 63, 63)"}}
            ))
            fig_waterfall.update_layout(
                title=f"Feature Contributions for {selected_ts}",
                yaxis_title="Contribution to Load (kWh)", height=350
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 6: WHAT-IF SCENARIO SIMULATOR
    # -------------------------------------------------------------
    elif menu == "🎛️ What-If Scenario Simulator":
        st.subheader("🎛️ Campus Operational What-If Scenario Simulator")
        
        c_ctrl1, c_ctrl2, c_ctrl3 = st.columns(3)
        with c_ctrl1:
            b_sim = st.selectbox("Simulation Target Building", list(building_profiles.keys()), format_func=lambda x: x.replace("_", " "))
            sim_horizon = st.slider("Simulation Horizon (Days)", 3, 14, 7)
        with c_ctrl2:
            temp_delta = st.slider("🌡️ Outdoor Temp Delta (°C) [Heatwave / Cold Snap]", -8.0, 10.0, 3.5, step=0.5)
            occupancy_multiplier = st.slider("👥 Occupancy Scaling Factor (Exam vs Vacation)", 0.2, 2.0, 1.25, step=0.05)
        with c_ctrl3:
            hvac_efficiency_gain = st.slider("❄️ HVAC Efficiency Improvement (%)", 0, 40, 10, step=5)
            exam_mode = st.checkbox("Toggle Campus Final Exam Week Active", value=True)

        sim_df = test_df[test_df["building_id"] == b_sim].tail(sim_horizon * 24).copy()
        
        sim_df_modified = sim_df.copy()
        sim_df_modified["temperature_c"] = np.clip(sim_df_modified["temperature_c"] + temp_delta, 0.0, 50.0)
        sim_df_modified["cooling_degree_hours"] = np.maximum(0.0, sim_df_modified["temperature_c"] - 22.0)
        sim_df_modified["occupancy_rate"] = np.clip(sim_df_modified["occupancy_rate"] * occupancy_multiplier, 0.0, 1.0)
        sim_df_modified["is_exam_week"] = 1 if exam_mode else 0
        sim_df_modified["occ_x_temp"] = sim_df_modified["occupancy_rate"] * sim_df_modified["temperature_c"]
        sim_df_modified["occ_x_exam"] = sim_df_modified["occupancy_rate"] * sim_df_modified["is_exam_week"]

        best_m = models[best_model_name]
        X_sim = sim_df_modified[artifacts["feature_cols"]]
        sim_preds = best_m.predict(X_sim)
        
        sim_preds_adjusted = sim_preds * (1.0 - (hvac_efficiency_gain / 100.0 * 0.40))
        sim_df["simulated_kwh"] = np.round(sim_preds_adjusted, 2)
        
        baseline_total = sim_df["forecast_kwh"].sum()
        sim_total = sim_df["simulated_kwh"].sum()
        pct_change = ((sim_total - baseline_total) / baseline_total) * 100.0
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Baseline Energy", f"{baseline_total:,.0f} kWh")
        col_s2.metric("Simulated Energy", f"{sim_total:,.0f} kWh", f"{pct_change:+.1f}%")
        col_s3.metric("Simulated Peak Load", f"{sim_df['simulated_kwh'].max():.1f} kWh", f"{sim_df['simulated_kwh'].max() - sim_df['forecast_kwh'].max():+.1f} kWh")
        col_s4.metric("Cost Differential", f"₹ {(sim_total - baseline_total)*8.5:,.0f}", "Tariff Impact")

        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(
            x=sim_df["timestamp"], y=sim_df["forecast_kwh"],
            mode="lines", name="Normal Baseline Load", line=dict(color="#64748B", width=2)
        ))
        fig_sim.add_trace(go.Scatter(
            x=sim_df["timestamp"], y=sim_df["simulated_kwh"],
            mode="lines", name="What-If Simulated Scenario", line=dict(color="#DC2626" if pct_change > 0 else "#10B981", width=2.5)
        ))
        fig_sim.update_layout(
            title=f"What-If Demand Projection ({b_sim.replace('_', ' ')})",
            xaxis_title="Time", yaxis_title="Load (kWh)", height=400
        )
        st.plotly_chart(fig_sim, use_container_width=True)

    # -------------------------------------------------------------
    # TAB 7: BUILDING BENCHMARKING & EUI
    # -------------------------------------------------------------
    elif menu == "🏛️ Building Benchmarks & EUI":
        st.subheader("🏛️ Multi-Building Energy Benchmarking & Efficiency Scorecard")
        
        eui_records = []
        for b_name, b_info in building_profiles.items():
            b_data = featured_df[featured_df["building_id"] == b_name]
            annual_kwh = b_data["total_power_kwh"].sum() / (len(b_data) / 8760.0)
            area = b_info["area_sqm"]
            eui = annual_kwh / area
            avg_p = b_data["total_power_kwh"].mean()
            peak_p = b_data["total_power_kwh"].max()
            par = peak_p / avg_p if avg_p > 0 else 1.0
            
            if eui < 70:
                rating = "⭐⭐⭐⭐⭐ (High Efficiency)"
            elif eui < 110:
                rating = "⭐⭐⭐⭐ (Good)"
            elif eui < 160:
                rating = "⭐⭐⭐ (Moderate)"
            else:
                rating = "⭐⭐ (High Consumption - Action Needed)"
                
            eui_records.append({
                "Building Name": b_name.replace("_", " "),
                "Type": b_info["type"],
                "Area (m²)": f"{area:,}",
                "Annual Power (kWh)": f"{annual_kwh:,.0f}",
                "EUI (kWh/m²/yr)": round(eui, 1),
                "Peak Load (kWh)": round(peak_p, 1),
                "Peak-to-Avg Ratio (PAR)": round(par, 2),
                "Sustainability Rating": rating
            })
            
        eui_df = pd.DataFrame(eui_records)
        st.dataframe(eui_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            fig_eui = px.bar(
                eui_df, x="Building Name", y="EUI (kWh/m²/yr)", color="Type",
                title="Energy Use Intensity Comparison (kWh / m² / year)",
                text="EUI (kWh/m²/yr)"
            )
            st.plotly_chart(fig_eui, use_container_width=True)
        with col_b2:
            fig_par = px.bar(
                eui_df, x="Building Name", y="Peak-to-Avg Ratio (PAR)", color="Type",
                title="Peak-to-Average Ratio (Grid Stress Metric)",
                text="Peak-to-Avg Ratio (PAR)"
            )
            st.plotly_chart(fig_par, use_container_width=True)


if __name__ == "__main__":
    main()
