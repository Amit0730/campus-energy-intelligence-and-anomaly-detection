"""One forecast per building and calendar day, issued before its first hour."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "day_ahead"
TARGET = "total_power_kwh"
ID = "building_id"
TIME = "timestamp"
CALENDAR = ["hour_sin", "hour_cos", "weekday", "weekend", "season_sin", "season_cos"]
CONTEXT = ["area_sqm", "temperature_yesterday", "occupancy_last_week"]
LAGS = ["yesterday", "last_week", "two_weeks_ago", "prior_day_mean"]
FEATURES = CALENDAR + CONTEXT + LAGS
DETECTORS = ["Residual rule", "Isolation Forest", "Combined"]
LEVELS = {"Sensitive": -0.025, "Standard": 0.0, "Conservative": 0.015}


class SeasonalAR:
    """SARIMA(0,0,0)(1,0,0)[24] with intercept, fitted by conditional least squares.

    Each of the five facilities has a distinct, constant floor area. For horizons
    1 through 24 the seasonal lag is available at the midnight forecast origin.
    """

    def __init__(self, parameters=None):
        self.parameters = parameters or {}

    def fit(self, x, y):
        for area, g in x.assign(target=np.asarray(y)).groupby("area_sqm"):
            lag = g.yesterday.to_numpy()
            phi, intercept = np.linalg.lstsq(
                np.column_stack([lag, np.ones(len(g))]), g.target, rcond=None
            )[0]
            self.parameters[str(int(area))] = [float(phi), float(intercept)]
        return self

    def predict(self, x):
        keys = x["area_sqm"].astype(int).astype(str)
        phi = keys.map({k: v[0] for k, v in self.parameters.items()})
        intercept = keys.map({k: v[1] for k, v in self.parameters.items()})
        return (phi * x.yesterday + intercept).to_numpy()


def design(group):
    """Input is one building. Reindexing keeps shifts in clock hours during gaps."""
    g = group.sort_values(TIME).set_index(TIME).asfreq("h")
    f = pd.DataFrame(index=g.index)
    f["hour_sin"] = np.sin(2 * np.pi * g.index.hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * g.index.hour / 24)
    f["weekday"] = g.index.dayofweek
    f["weekend"] = (g.index.dayofweek >= 5).astype(int)
    phase = 2 * np.pi * g.index.dayofyear / 365.25
    f["season_sin"], f["season_cos"] = np.sin(phase), np.cos(phase)
    f["area_sqm"] = g["area_sqm"].ffill()
    f["temperature_yesterday"] = g["temperature_c"].shift(24)
    f["occupancy_last_week"] = g["occupancy_rate"].shift(168)
    for col, lag in zip(LAGS[:3], [24, 168, 336]):
        f[col] = g[TARGET].shift(lag)
    daily = g[TARGET].resample("D").agg(["mean", "count"])
    f["prior_day_mean"] = f.index.normalize().map(
        daily["mean"].where(daily["count"].eq(24)).shift()
    )
    return f


def future_features(history):
    end = history[TIME].max()
    if end.hour != 23:
        raise ValueError("A complete final day is required for a midnight forecast.")
    future = pd.DataFrame(
        {TIME: pd.date_range(end + pd.Timedelta(hours=1), periods=24, freq="h")}
    )
    features = design(pd.concat([history, future], ignore_index=True)).iloc[-24:]
    if features[FEATURES].isna().any().any():
        raise ValueError("Required readings from the previous fortnight are missing.")
    return features[FEATURES]


def boost():
    return XGBRegressor(
        n_estimators=180,
        max_depth=5,
        learning_rate=0.075,
        objective="reg:squarederror",
        n_jobs=2,
        random_state=42,
        tree_method="hist",
    )


def errors(y, p):
    e = np.asarray(y) - np.asarray(p)
    return {
        "MAE": round(float(abs(e).mean()), 3),
        "RMSE": round(float(np.sqrt(np.mean(e**2))), 3),
    }


def confusion(labels, flags):
    y, f = np.asarray(labels, bool), np.asarray(flags, bool)
    tp, fp, fn = int((y & f).sum()), int((~y & f).sum()), int((y & ~f).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
        "Caught": tp,
        "False alarms": fp,
        "Missed": fn,
    }


def forecast(model, name, features):
    if name in ("Yesterday", "Last week"):
        return features["yesterday" if name == "Yesterday" else "last_week"].to_numpy()
    return np.maximum(0, model.predict(features[FEATURES]))


def load_predictor(out, summary):
    name = summary["winner"]
    if name == "Seasonal AR (24 h)":
        return SeasonalAR(summary["seasonal_parameters"])
    if name == "XGBoost":
        model = XGBRegressor()
        model.load_model(out / "forecast.json")
        return model
    if name == "Ridge":

        class Linear:
            def predict(self, x):
                return (
                    np.asarray(x) @ np.array(summary["coefficients"])
                    + summary["intercept"]
                )

        return Linear()
    return None


def rank_scores(values, reference):
    return np.searchsorted(np.sort(reference), values, side="right") / len(reference)


def build(out=OUTPUT):
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "raw_campus_energy_data.csv", parse_dates=[TIME])
    blocks = []
    for _, g in raw.groupby(ID):
        blocks.append(
            g.set_index(TIME).join(design(g).drop(columns="area_sqm")).reset_index()
        )
    data = (
        pd.concat(blocks, ignore_index=True).dropna(subset=FEATURES).sort_values(TIME)
    )
    days = np.sort(data[TIME].dt.normalize().unique())
    d1, d2, d3 = [pd.Timestamp(days[int(len(days) * p)]) for p in (0.60, 0.75, 0.85)]
    train = data[data[TIME] < d1]
    valid = data[data[TIME].between(d1, d2, inclusive="left")]
    cal = data[data[TIME].between(d2, d3, inclusive="left")]
    test = data[data[TIME] >= d3].copy()
    fitted = {
        "Yesterday": None,
        "Last week": None,
        "Seasonal AR (24 h)": SeasonalAR(),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
        "XGBoost": boost(),
    }
    val_metrics = {}
    for name, model in fitted.items():
        if model is not None:
            model.fit(train[FEATURES], train[TARGET])
        val_metrics[name] = errors(valid[TARGET], forecast(model, name, valid))
    winner = min(val_metrics, key=lambda n: val_metrics[n]["RMSE"])
    chosen = fitted[winner]
    summary = {
        "winner": winner,
        "features": FEATURES,
        "validation": val_metrics,
        "test": {},
        "dates": {
            n: [str(z[TIME].min().date()), str(z[TIME].max().date())]
            for n, z in [
                ("Train", train),
                ("Selection", valid),
                ("Calibration", cal),
                ("Test", test),
            ]
        },
        "counts": {
            n: len(z)
            for n, z in [
                ("Train", train),
                ("Selection", valid),
                ("Calibration", cal),
                ("Test", test),
            ]
        },
    }
    if winner == "XGBoost":
        chosen.save_model(out / "forecast.json")
    elif winner == "Seasonal AR (24 h)":
        summary["seasonal_parameters"] = chosen.parameters
    elif winner == "Ridge":
        scale, reg = chosen.steps[0][1], chosen.steps[1][1]
        coef = reg.coef_ / scale.scale_
        summary.update(
            coefficients=coef.tolist(),
            intercept=float(reg.intercept_ - scale.mean_ @ coef),
        )
    for name, model in fitted.items():
        summary["test"][name] = errors(test[TARGET], forecast(model, name, test))
    summary["ablation"] = {"Full XGBoost": summary["test"]["XGBoost"]}
    for name, drop in [
        ("Without weather", ["temperature_yesterday"]),
        ("Without occupancy", ["occupancy_last_week"]),
        ("Without past energy", LAGS),
    ]:
        cols = [c for c in FEATURES if c not in drop]
        m = boost().fit(train[cols], train[TARGET])
        summary["ablation"][name] = errors(
            test[TARGET], np.maximum(0, m.predict(test[cols]))
        )
    valid = valid.copy()
    cal = cal.copy()
    for z in [valid, cal, test]:
        z["expected"] = forecast(chosen, winner, z)
        z["residual"] = z[TARGET] - z["expected"]
    # Fixed model, independent calibration period, symmetric 80% interval per facility.
    widths = cal.groupby(ID)["residual"].apply(
        lambda s: float(np.quantile(s.abs(), 0.8, method="higher"))
    )
    summary["widths"] = widths.to_dict()
    test["low"] = (test.expected - test[ID].map(widths)).clip(lower=0)
    test["high"] = test.expected + test[ID].map(widths)
    test["covered"] = test[TARGET].between(test.low, test.high)
    summary["coverage"] = float(test.covered.mean())
    summary["building_metrics"] = {
        b: dict(**errors(g[TARGET], g.expected), coverage=float(g.covered.mean()))
        for b, g in test.groupby(ID)
    }
    # Fit anomaly context on earlier forecast errors, freeze thresholds before test.
    thresholds = {}
    for b, history in valid.groupby(ID):
        calibration = cal[cal[ID] == b]
        mask = test[ID] == b
        detector_cols = [
            "residual",
            TARGET,
            "occupancy_rate",
            "hour_sin",
            "hour_cos",
            "temperature_c",
        ]
        forest = IsolationForest(n_estimators=80, random_state=42, n_jobs=2).fit(
            history[detector_cols]
        )
        reference = [
            history.residual.abs(),
            -forest.score_samples(history[detector_cols]),
        ]

        def scores(frame):
            r = rank_scores(frame.residual.abs(), reference[0])
            i = rank_scores(-forest.score_samples(frame[detector_cols]), reference[1])
            return [r, i, (r + i) / 2]

        thresholds[b] = {}
        for method, c_score, t_score in zip(
            DETECTORS, scores(calibration), scores(test[mask])
        ):
            quantiles = np.linspace(0.88, 0.995, 47)

            def calibration_f1(q):
                flags = c_score >= np.quantile(c_score, q)
                return confusion(calibration.is_anomaly, flags)["F1"]

            best = max(quantiles, key=calibration_f1)
            thresholds[b][method] = {
                level: float(np.quantile(c_score, np.clip(best + offset, 0, 1)))
                for level, offset in LEVELS.items()
            }
            test.loc[mask, method] = t_score
    summary["thresholds"] = thresholds
    keep = [
        TIME,
        ID,
        "area_sqm",
        "temperature_c",
        "occupancy_rate",
        TARGET,
        "is_anomaly",
        "anomaly_type",
        "expected",
        "residual",
        "low",
        "high",
    ] + DETECTORS
    summary["detectors"] = {}
    for method in DETECTORS:
        cutoffs = {b: v[method]["Standard"] for b, v in thresholds.items()}
        summary["detectors"][method] = confusion(
            test.is_anomaly, test[method] >= test[ID].map(cutoffs)
        )
    test[keep].sort_values([ID, TIME]).to_csv(
        out / "evaluation.csv.gz",
        index=False,
        compression={"method": "gzip", "mtime": 0},
        float_format="%.17g",
    )
    (out / "experiment.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(winner, summary["test"], "coverage", summary["coverage"])


if __name__ == "__main__":
    build()
