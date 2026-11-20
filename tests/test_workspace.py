import json
from pathlib import Path
import tempfile
import unittest
import os
import pandas as pd
import numpy as np
from streamlit.testing.v1 import AppTest
from src.day_ahead import (
    ROOT,
    OUTPUT,
    TIME,
    ID,
    TARGET,
    FEATURES,
    design,
    future_features,
    forecast,
    load_predictor,
    errors,
)
from src.incidents import group_incidents, reviews, save_review


class EnergyWorkspaceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(
            ROOT / "data" / "raw_campus_energy_data.csv", parse_dates=[TIME]
        )
        cls.s = json.loads((OUTPUT / "experiment.json").read_text())
        cls.rows = pd.read_csv(OUTPUT / "evaluation.csv.gz", parse_dates=[TIME])

    def test_forecast_matches_historical_issue_with_unseen_day_removed(self):
        building = self.raw[ID].iloc[0]
        raw = self.raw[self.raw[ID] == building]
        day = pd.Timestamp("2025-09-25")
        xs = future_features(raw[raw[TIME] < day])
        np.testing.assert_allclose(
            xs, design(raw).loc[day : day + pd.Timedelta(hours=23), FEATURES]
        )
        expected = self.rows[
            (self.rows[ID] == building) & self.rows[TIME].dt.normalize().eq(day)
        ].expected
        np.testing.assert_allclose(
            forecast(load_predictor(OUTPUT, self.s), self.s["winner"], xs),
            expected,
            rtol=1e-6,
        )

    def test_summary_and_incidents(self):
        self.assertEqual(
            errors(self.rows[TARGET], self.rows.expected),
            self.s["test"][self.s["winner"]],
        )
        self.assertEqual(
            self.s["winner"],
            min(self.s["validation"], key=lambda k: self.s["validation"][k]["RMSE"]),
        )
        events = group_incidents(self.rows, self.s)
        self.assertTrue(events.event_id.is_unique)
        self.assertTrue(
            (
                events.hours
                == (events.end - events.start).dt.total_seconds() / 3600 + 1
            ).all()
        )
        self.assertTrue(
            (
                (events.start.dt.date <= pd.Timestamp("2025-09-25").date())
                & (events.end.dt.date >= pd.Timestamp("2025-09-25").date())
            ).any()
        )

    def test_review_persistence_and_workspace_isolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reviews.sqlite"
            self.assertTrue(
                save_review(path, "A", "event", "Investigating", "check HVAC", 0)
            )
            self.assertFalse(
                save_review(path, "A", "event", "Resolved", "stale edit", 0)
            )
            self.assertEqual(
                reviews(path, "A")["event"], ("Investigating", "check HVAC", 1)
            )
            self.assertFalse(reviews(path, "B"))

    def test_ui_journey(self):
        with tempfile.TemporaryDirectory() as tmp:
            from unittest.mock import patch

            with patch.dict(
                os.environ, ENERGY_REVIEW_DB=str(Path(tmp) / "reviews.sqlite")
            ):
                at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
                self.assertFalse(at.exception)
                button = next(b for b in at.button if b.label == "Investigate")
                button.click().run()
                self.assertFalse(at.exception)
                self.assertTrue(any(t.value == "Incident desk" for t in at.title))
                selected = at.selectbox(key="event_key").value
                next(
                    s for s in at.selectbox if s.label == "Investigation status"
                ).set_value("Investigating")
                at.text_area[0].input("Verify night schedule.")
                next(b for b in at.button if b.label == "Save review").click().run()
                self.assertFalse(at.exception)
                self.assertTrue(
                    any(
                        "Download inspection report" == d.label
                        for d in at.get("download_button")
                    )
                )
                fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
                fresh.query_params["review"] = at.query_params["review"]
                fresh.run()
                fresh.radio(key="page").set_value("Incident desk").run()
                fresh.selectbox(key="event_key").set_value(selected).run()
                self.assertEqual(fresh.text_area[0].value, "Verify night schedule.")
                fresh.radio(key="page").set_value("Building explorer").run()
                self.assertFalse(fresh.exception)
                fresh.radio(key="page").set_value("Forecast lab").run()
                next(
                    b for b in fresh.button if b.label == "Forecast next 24 hours"
                ).click().run()
                self.assertFalse(fresh.exception)
                self.assertTrue(
                    any(
                        d.label == "Download 24-hour forecast"
                        for d in fresh.get("download_button")
                    )
                )


if __name__ == "__main__":
    unittest.main()
