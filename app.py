"""Campus Energy Intelligence, the Streamlit entry point."""

import json
import os
from datetime import timedelta
from html import escape
from pathlib import Path
import secrets
import re
import sqlite3
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from src.day_ahead import (
    ROOT,
    OUTPUT,
    ID,
    TIME,
    TARGET,
    LEVELS,
    DETECTORS,
    forecast,
    load_predictor,
    future_features,
    confusion,
)
from src.incidents import (
    STATES,
    detect,
    group_incidents,
    investigation,
    reviews,
    save_review,
    inspection_report,
)

PAGES = ["Campus overview", "Building explorer", "Forecast lab", "Incident desk"]
DB = Path(os.environ.get("ENERGY_REVIEW_DB", str(ROOT / ".local" / "reviews.sqlite")))
GREEN = "#007b68"
INK = "#203a4a"
AMBER = "#bd5819"


@st.cache_data
def load_data():
    experiment = json.loads((OUTPUT / "experiment.json").read_text())
    readings = pd.read_csv(
        OUTPUT / "evaluation.csv.gz", parse_dates=[TIME], float_precision="round_trip"
    )
    raw = pd.read_csv(ROOT / "data" / "raw_campus_energy_data.csv", parse_dates=[TIME])
    return experiment, readings, raw, group_incidents(readings, experiment)


@st.cache_resource
def predictor():
    return load_predictor(OUTPUT, json.loads((OUTPUT / "experiment.json").read_text()))


def friendly(name):
    return name.replace("_", " ")


def display_metrics(items):
    cards = "".join(
        f'<div class="stat"><span>{escape(label)}</span><strong>{escape(value)}</strong><small>{escape(note)}</small></div>'
        for label, value, note in items
    )
    st.markdown('<div class="stat-row">' + cards + "</div>", unsafe_allow_html=True)


def styles():
    st.markdown(
        """<style>
    [data-testid="stMainBlockContainer"]{max-width:1300px;padding-top:3.7rem}
    [data-testid="stSidebar"]{border-right:1px solid #d6e1df}
    .brand{display:flex;align-items:center;gap:12px;font-size:1.22rem;font-weight:750;color:#173d39;margin:4px 0 24px}
    .brand i{font-size:1.45rem;color:white;background:#007b68;border-radius:10px;padding:0 10px;font-style:normal}
    h1,h2,h3{letter-spacing:-.035em} h1{font-size:clamp(2rem,3.2vw,3.25rem)!important}
    .stat-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,180px),1fr));border:1px solid #cfddda;background:#f1f7f5;border-radius:14px;overflow:hidden;margin:0 0 .55rem}
    .stat{display:flex;flex-direction:column;padding:18px 20px;gap:4px}.stat span{font-size:.84rem;font-weight:650;color:#34534f}
    .stat strong{font-size:clamp(1.3rem,2vw,2rem);letter-spacing:-.04em;color:#102f2a;line-height:1.2}
    .stat small{color:#49605c;font-size:.78rem}.source{color:#43655d;font-size:.83rem;font-weight:650}
    </style>""",
        unsafe_allow_html=True,
    )


def series_chart(df, band=False, selected=None):
    fig = go.Figure()
    if band:
        fig.add_scatter(
            x=df[TIME],
            y=df.high,
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
        fig.add_scatter(
            x=df[TIME],
            y=df.low,
            fill="tonexty",
            fillcolor="rgba(0,123,104,.13)",
            line=dict(width=0),
            name="80% target range",
            hoverinfo="skip",
        )
    if TARGET in df:
        fig.add_scatter(
            x=df[TIME], y=df[TARGET], name="Actual", line=dict(color=INK, width=2.1)
        )
    fig.add_scatter(
        x=df[TIME],
        y=df.expected,
        name="Forecast",
        line=dict(color=GREEN, width=2, dash="dash"),
    )
    if selected is not None:
        fig.add_scatter(
            x=selected[TIME],
            y=selected[TARGET],
            name="Flagged hour",
            mode="markers",
            marker=dict(size=10, color=AMBER),
        )
    fig.update_layout(
        height=335,
        margin=dict(l=4, r=12, t=22, b=6),
        legend=dict(orientation="h", y=1.13),
        yaxis_title="Electricity (kWh)",
        hovermode="x unified",
        xaxis_title=None,
    )
    st.plotly_chart(fig, width="stretch")


def workspace_key():
    key = st.query_params.get("review", "")
    if not re.fullmatch(r"[0-9a-f]{32}", key):
        key = secrets.token_hex(16)
        st.query_params["review"] = key
    return key


def open_incident(key):
    st.session_state.page = "Incident desk"
    st.session_state.event_key = key
    st.session_state.review_filter = "All statuses"
    st.session_state.case_search = ""


def open_example(event):
    st.session_state.period = (
        event.start.date() - timedelta(days=1),
        event.end.date() + timedelta(days=1),
    )
    st.session_state.facility = event.building
    open_incident(event.event_id)


def overview(rows, events, notes):
    st.title("Campus overview")
    hourly = rows.groupby(TIME)[[TARGET, "expected"]].sum().reset_index()
    open_events = events[
        events.event_id.map(lambda k: notes.get(k, ("Open",))[0] == "Open")
    ]
    peak = hourly.loc[hourly[TARGET].idxmax()]
    display_metrics(
        [
            (
                "Energy consumed",
                f"{rows[TARGET].sum() / 1000:,.1f} MWh",
                "Selected buildings and dates",
            ),
            (
                "Highest hourly use",
                f"{peak[TARGET]:,.0f} kWh",
                peak[TIME].strftime("%d %b · %H:%M"),
            ),
            ("Open incidents", str(len(open_events)), "Awaiting a first review"),
        ]
    )
    if not open_events.empty:
        first = open_events.iloc[0]
        with st.container(border=True):
            evidence, action = st.columns([3, 1], vertical_alignment="center")
            evidence.markdown(
                f"**First to inspect · {friendly(first.building)}**  \n{first.start:%d %b, %H:%M} · {first.difference:+,.1f} kWh across {first.hours} flagged hour(s)"
            )
            action.button(
                "Open incident",
                type="primary",
                on_click=open_incident,
                args=(first.event_id,),
            )
    st.subheader("Actual and expected use")
    # For long periods daily totals keep the chart legible.
    if len(hourly) > 24 * 15:
        hourly = hourly.set_index(TIME).resample("D").sum().reset_index()
        st.caption("Daily totals for the selected period.")
    series_chart(hourly)
    st.subheader("Priority incidents")
    st.caption(
        "Open investigations ranked by absolute difference from forecast. Consecutive flags in the same direction form one incident."
    )
    if open_events.empty:
        st.info("All incidents in this view have been reviewed.")
    for e in open_events.head(3).itertuples():
        with st.container(border=True):
            a, b, c = st.columns([3, 2, 1.3], vertical_alignment="center")
            a.markdown(f"**{friendly(e.building)}**  \n{e.start:%d %b %Y · %H:%M}")
            b.write(f"{e.difference:+,.1f} kWh · {e.hours} h")
            c.button(
                "Investigate",
                key="open_" + e.event_id,
                on_click=open_incident,
                args=(e.event_id,),
            )
    st.subheader("Building comparison")
    table = (
        rows.groupby(ID)
        .agg(energy=(TARGET, "sum"), area=("area_sqm", "first"), hours=(TIME, "size"))
        .reset_index()
    )
    table["kWh/m²"] = (table.energy / table.area).round(3)
    table["Flagged hours (%)"] = (
        (100 * rows.groupby(ID).flagged.mean()).reindex(table[ID]).to_numpy().round(2)
    )
    # Empirical ranking scaled to the displayed alert rate rather than an undocumented probability.
    table = table.sort_values("Flagged hours (%)", ascending=False)
    table[ID] = table[ID].map(friendly)
    st.dataframe(
        table[[ID, "energy", "kWh/m²", "Flagged hours (%)"]].rename(
            columns={ID: "Building", "energy": "Total kWh"}
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Risk indicator = flagged readings ÷ observed readings. Energy intensity uses floor area and the dates selected above."
    )


def choose_building(buildings, scope, key):
    if scope != "All buildings":
        st.markdown(f"**{friendly(scope)}**")
        return scope
    return st.selectbox("Building", buildings, format_func=friendly, key=key)


def explorer(raw, rows, building, start, end):
    st.title("Building explorer")
    g = raw[
        (raw[ID] == building) & raw[TIME].between(start, end, inclusive="left")
    ].copy()
    st.caption(
        f"{friendly(building)} · {g.building_type.iloc[0]} · {g.area_sqm.iloc[0]:,.0f} m²"
    )
    display_metrics(
        [
            (
                "Use per floor area",
                f"{g[TARGET].sum() / g.area_sqm.iloc[0]:.2f} kWh/m²",
                "Selected period",
            ),
            (
                "Busiest hour",
                g.loc[g[TARGET].idxmax(), TIME].strftime("%d %b · %H:%M"),
                f"{g[TARGET].max():,.1f} kWh",
            ),
            ("Mean occupancy", f"{g.occupancy_rate.mean():.0%}", "Simulated occupancy"),
        ]
    )
    st.subheader("When does this building use electricity?")
    matrix = g.pivot_table(
        index="day_of_week", columns="hour", values=TARGET, aggfunc="mean"
    ).reindex(index=range(7), columns=range(24))
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(range(24)),
            y=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            colorscale=[[0, "#eff7f3"], [0.5, "#73bb9d"], [1, "#075745"]],
            colorbar=dict(title="kWh"),
            hovertemplate="%{y} %{x}:00 · %{z:.1f} kWh<extra></extra>",
        )
    )
    fig.update_layout(
        yaxis_autorange="reversed",
        height=290,
        xaxis_title="Hour of day",
        margin=dict(t=8, l=8, b=8, r=8),
    )
    st.plotly_chart(fig, width="stretch")
    cycles = (
        g.groupby(["hour", "is_weekend"])[TARGET]
        .mean()
        .unstack()
        .rename(columns={0: "Weekday", 1: "Weekend"})
    )
    st.subheader("Weekday and weekend profiles")
    st.line_chart(cycles, x_label="Hour of day", y_label="Mean hourly kWh")
    with st.expander("Meter breakdown and data quality"):
        st.bar_chart(
            g.groupby("hour")[
                ["hvac_power_kwh", "lighting_power_kwh", "equipment_power_kwh"]
            ].mean(),
            x_label="Hour",
            y_label="Mean hourly kWh",
        )
        st.write(
            f"{len(g):,} hourly readings · {g[TIME].duplicated().sum()} duplicate timestamps · {g[TARGET].isna().sum()} missing energy values. End-use channels are simulated."
        )
    st.download_button(
        "Download building readings",
        g.to_csv(index=False),
        "building-readings.csv",
        "text/csv",
    )


def table_metrics(records, label):
    f = pd.DataFrame.from_dict(records, orient="index").rename_axis(label).reset_index()
    st.dataframe(f, hide_index=True, width="stretch")


def forecast_lab(raw, building, s):
    st.title("Forecast lab")
    g = raw[raw[ID] == building].sort_values(TIME)
    st.write(
        "Predict the next 24 hourly readings from the end of the selected building’s history."
    )
    st.caption(
        f"{friendly(building)} · last observation {g[TIME].max():%d %b %Y, %H:%M}. Weather uses yesterday’s same-hour reading. Occupancy uses last week’s same hour."
    )
    if st.button("Forecast next 24 hours", type="primary"):
        st.session_state.forecast_building = building
    if st.session_state.get("forecast_building") == building:
        features = future_features(g)
        point = forecast(predictor(), s["winner"], features)
        w = s["widths"][building]
        future = pd.DataFrame(
            {
                TIME: features.index,
                "expected": point,
                "low": np.maximum(0, point - w),
                "high": point + w,
            }
        )
        peak = future.loc[future.expected.idxmax()]
        display_metrics(
            [
                ("Forecast day", future[TIME].min().strftime("%d %b %Y"), s["winner"]),
                ("Expected total", f"{point.sum():,.0f} kWh", "24 hourly predictions"),
                (
                    "Expected peak",
                    peak[TIME].strftime("%H:%M"),
                    f"{peak.expected:,.1f} kWh in that hour",
                ),
            ]
        )
        series_chart(future, band=True)
        st.download_button(
            "Download 24-hour forecast",
            future.to_csv(index=False, float_format="%.3f"),
            "day-ahead-forecast.csv",
            "text/csv",
        )
        if s["winner"] == "XGBoost":
            with st.expander("What drove the peak prediction?"):
                import xgboost as xgb

                contrib = (
                    predictor()
                    .get_booster()
                    .predict(
                        xgb.DMatrix(features.loc[[peak[TIME]]]), pred_contribs=True
                    )[0]
                )
                terms = pd.Series(contrib[:-1], index=features.columns)
                st.caption(
                    f"Base prediction {contrib[-1]:.1f} kWh. Largest model contributions for this hour, in kWh."
                )
                st.bar_chart(
                    terms.loc[terms.abs().nlargest(5).index].rename(
                        "Contribution (kWh)"
                    ),
                    horizontal=True,
                )
        with st.expander("Estimate the effect of shifting peak use"):
            fraction = st.slider(
                "Share of peak-hour energy shifted to the lowest-use hour (%)",
                0,
                40,
                10,
            )
            adjusted = point.astype(float).copy()
            amount = point.max() * fraction / 100
            adjusted[np.argmax(point)] -= amount
            adjusted[np.argmin(point)] += amount
            st.write(
                f"Shifting {amount:.1f} kWh would change the highest hourly use from {point.max():.1f} to {adjusted.max():.1f} kWh. Total daily energy stays at {adjusted.sum():,.1f} kWh."
            )
    st.subheader("Model comparison")
    st.caption(
        "Each model forecasts the same 24 hours at midnight. Selection uses the lowest validation RMSE. Errors below are kWh per building-hour."
    )
    results = pd.DataFrame(s["test"]).T.join(
        pd.Series(
            {k: v["RMSE"] for k, v in s["validation"].items()}, name="Validation RMSE"
        )
    )
    results.index.name = "Model"
    st.dataframe(
        results.rename(columns={"MAE": "Test MAE", "RMSE": "Test RMSE"}).reset_index(),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        " · ".join(f"{n} {dates[0]} to {dates[1]}" for n, dates in s["dates"].items())
    )
    with st.expander("Feature experiments and uncertainty"):
        table_metrics(s["ablation"], "XGBoost feature set")
        st.write(
            f"The range targets 80% coverage using each building’s calibration errors. It contained {s['coverage']:.1%} of held-out hourly readings. Wider ranges cover more observations."
        )
        table_metrics(
            {
                friendly(k): {
                    "MAE": v["MAE"],
                    "RMSE": v["RMSE"],
                    "Range coverage (%)": round(100 * v["coverage"], 1),
                }
                for k, v in s["building_metrics"].items()
            },
            "Building",
        )
    real = json.loads((OUTPUT / "real_meter.json").read_text())
    with st.expander("Independent test on a measured education-building meter"):
        st.write(
            f"{real['building']} · Building Data Genome 2. Trained on {real['train_year']} meter readings and tested on {real['test_year']} ({real['test_hours']:,} usable hours)."
        )
        st.caption(
            "A separate model uses calendar, yesterday and last week. Hours with missing targets or required lags are excluded."
        )
        table_metrics(real["scores"], "Model")
        st.markdown(
            "[Meter data and attribution](https://github.com/buds-lab/building-data-genome-project-2)"
        )
    st.download_button(
        "Download experiment evidence",
        (OUTPUT / "experiment.json").read_bytes(),
        "experiment.json",
        "application/json",
    )


def incident_desk(rows, full, events, s, workspace, notes, tariff, db_available):
    st.title("Incident desk")
    st.caption(
        "Reviews reopen at the same workspace link. Download a report to keep a copy across hosted app restarts."
    )
    a, b = st.columns([2, 1])
    search = (
        a.text_input(
            "Search building or date",
            key="case_search",
            placeholder="Academic Block or 2025-09-25",
        )
        .strip()
        .casefold()
    )
    state = b.selectbox("Review status", ["All statuses", *STATES], key="review_filter")
    cases = events.copy()
    cases["state"] = cases.event_id.map(lambda k: notes.get(k, ("Open",))[0])
    if state != "All statuses":
        cases = cases[cases.state == state]
    if search:
        combined = cases.building.map(friendly) + " " + cases.start.astype(str)
        cases = cases[combined.str.casefold().str.contains(search, regex=False)]
    if cases.empty:
        st.info("No incidents match these filters.")
    else:
        labels = {
            e.event_id: f"{friendly(e.building)} · {e.start:%d %b %Y, %H:%M} · {e.difference:+.1f} kWh"
            for e in cases.itertuples()
        }
        if st.session_state.get("event_key") not in labels:
            st.session_state.event_key = next(iter(labels))
        key = st.selectbox(
            "Incident", list(labels), format_func=labels.get, key="event_key"
        )
        e = cases.loc[cases.event_id == key].iloc[0]
        record = full[(full[ID] == e.building) & full[TIME].between(e.start, e.end)]
        context = full[
            (full[ID] == e.building)
            & full[TIME].between(
                e.start - pd.Timedelta(hours=12), e.end + pd.Timedelta(hours=12)
            )
        ]
        st.subheader(friendly(e.building))
        st.caption(
            f"{e.start:%d %b %Y, %H:%M} through {e.end:%d %b %Y, %H:%M} · {e.hours} flagged hour(s) · {e.direction}"
        )
        display_metrics(
            [
                ("Actual", f"{e.actual:,.1f} kWh", "Across flagged hours"),
                ("Expected", f"{e.expected:,.1f} kWh", "Day-ahead forecast"),
                (
                    "Difference",
                    f"{e.difference:+,.1f} kWh",
                    f"Mean occupancy {e.occupancy:.0%}",
                ),
            ]
        )
        series_chart(context, band=True, selected=record)
        st.write(f"**Suggested inspection**  \n{investigation(e)}")
        st.caption(
            f"Estimated excess cost ₹{max(0, e.difference) * tariff:,.2f} at the selected ₹{tariff:g}/kWh tariff. Actual savings require verification."
        )
        status, note, revision = notes.get(key, ("Open", "", 0))
        form_id = f"{key}_{revision}"
        with st.form("review_" + form_id):
            change = st.selectbox(
                "Investigation status",
                STATES,
                index=STATES.index(status),
                key="status_" + form_id,
            )
            new_note = st.text_area(
                "Inspection note",
                value=note,
                max_chars=2000,
                key="note_" + form_id,
                placeholder="Record what was checked and what to inspect next.",
            )
            saved = st.form_submit_button("Save review", disabled=not db_available)
        if saved:
            try:
                if save_review(DB, workspace, key, change, new_note, revision):
                    st.session_state.review_message = "Review saved."
                else:
                    st.session_state.review_message = "This review changed in another tab. Read the latest version and try again."
                st.rerun()
            except sqlite3.Error:
                st.error("Could not save the review. Retry or download your draft.")
        st.download_button(
            "Download inspection report",
            inspection_report(e, change, new_note, record, tariff),
            "inspection-report.txt",
            "text/plain",
        )
    with st.expander("Test the detectors and inspect missed cases"):
        level = st.select_slider(
            "Detection sensitivity",
            options=["Conservative", "Standard", "Sensitive"],
            value="Standard",
        )
        flags = {m: detect(rows, s, m, level) for m in DETECTORS}
        table_metrics(
            {m: confusion(rows.is_anomaly, f) for m, f in flags.items()}, "Detector"
        )
        st.caption(
            "Compared with labelled synthetic faults in this selection. Standard thresholds maximize F1 on earlier calibration data per building. Changing sensitivity here leaves saved investigations at the standard setting."
        )
        missed = rows[(rows.is_anomaly == 1) & ~flags["Combined"]]
        if not missed.empty:
            miss = missed.iloc[missed.residual.abs().argmin()]
            limit = s["thresholds"][miss[ID]]["Combined"][level]
            st.write(
                f"**Missed example** · {friendly(miss[ID])}, {miss[TIME]:%d %b %Y, %H:%M}. The simulator injected “{miss.anomaly_type}”. Actual {miss[TARGET]:.1f} kWh, expected {miss.expected:.1f} kWh. The combined score {miss['Combined']:.4f} was below this building’s {limit:.4f} cutoff."
            )
        else:
            st.write("No missed labelled hours in this selection.")


def main():
    st.set_page_config(
        page_title="Campus Energy Intelligence", page_icon="⚡", layout="wide"
    )
    styles()
    s, full, raw, all_events = load_data()
    buildings = sorted(full[ID].unique())
    with st.sidebar:
        st.markdown(
            '<div class="brand"><i>ϟ</i> Campus Energy</div>', unsafe_allow_html=True
        )
        page = st.radio("Workspace", PAGES, key="page")
        st.divider()
        scope = st.selectbox(
            "Facility",
            ["All buildings", *buildings],
            format_func=friendly,
            key="facility",
        )
        latest = full[TIME].max().date()
        period = st.date_input(
            "Historical review dates",
            (latest - timedelta(days=13), latest),
            min_value=full[TIME].min().date(),
            max_value=latest,
            key="period",
        )
        if len(period) != 2:
            st.info("Choose an end date.")
            st.stop()
        tariff = (
            st.number_input(
                "Estimated tariff (₹/kWh)", min_value=0.0, value=8.50, step=0.5
            )
            if page == "Incident desk"
            else 8.5
        )
        st.divider()
        st.markdown("**Data source**  \nSimulated campus · 5 facilities")
        st.caption("Hourly readings · Jan 2024 – Dec 2025")
        with st.expander("Project details"):
            st.write(
                "Campus Energy Consumption Intelligence & Anomaly Detection · INT395. Amit Kumar Singh and Atharva Rotake."
            )
            st.write(
                "The forecast lab also includes an independent measured-meter benchmark."
            )
            st.markdown(
                "[Project repository](https://github.com/Amit0730/campus-energy-intelligence-and-anomaly-detection)"
            )
            instant = pd.Timestamp("2025-09-25 02:00")
            demo = all_events[
                (all_events.building == "Academic_Block_1")
                & (all_events.start <= instant)
                & (all_events.end >= instant)
            ]
            if not demo.empty:
                st.button(
                    "Open the overnight case study",
                    on_click=open_example,
                    args=(demo.iloc[0],),
                )
            for label, file, kind in [
                (
                    "Download project report",
                    "Campus_Energy_Report.pdf",
                    "application/pdf",
                ),
                (
                    "Download viva slides",
                    "Campus_Energy_Viva.pptx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ),
            ]:
                st.download_button(
                    label, (OUTPUT / "assessment" / file).read_bytes(), file, kind
                )
    start = pd.Timestamp(period[0])
    end = pd.Timestamp(period[1]) + pd.Timedelta(days=1)
    rows = full[full[TIME].between(start, end, inclusive="left")].copy()
    events = all_events[(all_events.start < end) & (all_events.end >= start)]
    if scope != "All buildings":
        rows = rows[rows[ID] == scope]
        events = events[events.building == scope]
    rows["flagged"] = detect(rows, s)
    if page in ["Campus overview", "Incident desk"]:
        workspace = workspace_key()
        try:
            notes = reviews(DB, workspace)
            available = True
        except (OSError, sqlite3.Error):
            st.warning(
                "Review storage is unavailable. You can still inspect and download reports."
            )
            notes, available = {}, False
        if msg := st.session_state.pop("review_message", None):
            st.info(msg)
        if page == "Campus overview":
            overview(rows, events, notes)
        else:
            incident_desk(rows, full, events, s, workspace, notes, tariff, available)
    else:
        building = choose_building(buildings, scope, "detail_building")
        if page == "Building explorer":
            explorer(raw, rows, building, start, end)
        else:
            forecast_lab(raw, building, s)


if __name__ == "__main__":
    main()
