"""Translate flagged readings into investigations and save review decisions."""

from contextlib import closing
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from .day_ahead import ID, TIME, TARGET

STATES = ("Open", "Investigating", "Expected activity", "Resolved")


def detect(data, experiment, method="Combined", level="Standard"):
    limits = {b: v[method][level] for b, v in experiment["thresholds"].items()}
    return data[method] >= data[ID].map(limits)


def group_incidents(data, experiment):
    flagged = data[detect(data, experiment)].sort_values([ID, TIME]).copy()
    flagged["direction"] = np.where(
        flagged.residual >= 0, "Above forecast", "Below forecast"
    )
    starts = (
        flagged[ID].ne(flagged[ID].shift())
        | flagged[TIME].diff().ne(pd.Timedelta(hours=1))
        | flagged.direction.ne(flagged.direction.shift())
    )
    events = flagged.groupby(starts.cumsum()).agg(
        building=(ID, "first"),
        start=(TIME, "min"),
        end=(TIME, "max"),
        hours=(TIME, "size"),
        actual=(TARGET, "sum"),
        expected=("expected", "sum"),
        occupancy=("occupancy_rate", "mean"),
        direction=("direction", "first"),
        peak_deviation=("residual", lambda v: v.abs().max()),
    )
    events["difference"] = events.actual - events.expected
    events["priority"] = events.difference.abs()
    events["event_id"] = [
        f"{b}/{start:%Y%m%d%H}" for b, start in zip(events.building, events.start)
    ]
    return events.sort_values("priority", ascending=False).reset_index(drop=True)


def investigation(event):
    if event.direction == "Below forecast":
        return "Check meter continuity and the building closure schedule."
    if event.occupancy < 0.15 and (event.start.hour < 7 or event.start.hour >= 21):
        return "Check overnight HVAC and lighting schedules against the occupancy log."
    return "Check scheduled activities, occupancy and equipment submeter readings for this period."


def connect_reviews(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=8)
    db.execute("""CREATE TABLE IF NOT EXISTS reviews
        (workspace TEXT, event TEXT, status TEXT, note TEXT, revision INTEGER,
         PRIMARY KEY(workspace,event))""")
    return db


def reviews(path, workspace):
    with closing(connect_reviews(path)) as db:
        return {
            e: (s, n, r)
            for e, s, n, r in db.execute(
                "SELECT event,status,note,revision FROM reviews WHERE workspace=?",
                (workspace,),
            )
        }


def save_review(path, workspace, event, status, note, revision):
    if status not in STATES or len(note) > 2000:
        raise ValueError("Choose a listed status and use at most 2,000 characters.")
    with closing(connect_reviews(path)) as db, db:
        if revision == 0:
            cursor = db.execute(
                "INSERT OR IGNORE INTO reviews VALUES(?,?,?,?,1)",
                (workspace, event, status, note),
            )
        else:
            cursor = db.execute(
                """UPDATE reviews SET status=?,note=?,revision=revision+1
                WHERE workspace=? AND event=? AND revision=?""",
                (status, note, workspace, event, revision),
            )
        return cursor.rowcount == 1


def inspection_report(event, status, note, records, tariff):
    return (
        f"Campus energy inspection report\nSimulated campus • day-ahead forecast\n\n"
        f"Building: {event.building.replace('_', ' ')}\nFirst reading: {event.start}\nLast reading: {event.end}\n"
        f"Flagged hours: {event.hours}\nActual: {event.actual:.2f} kWh\nExpected: {event.expected:.2f} kWh\n"
        f"Difference: {event.difference:+.2f} kWh\nEstimated excess cost at INR {tariff:.2f}/kWh: "
        f"INR {max(0, event.difference) * tariff:.2f}\nAverage occupancy: {event.occupancy:.1%}\n"
        f"Suggested inspection: {investigation(event)}\nReview status: {status}\nReview note: {note or '—'}\n\n"
        + records[[TIME, TARGET, "expected", "low", "high"]]
        .to_csv(index=False, float_format="%.2f")
    )
