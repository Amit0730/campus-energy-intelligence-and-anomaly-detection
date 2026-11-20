"""Reproduce the independent BDG2 forecast on one measured education meter."""

import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .day_ahead import boost, errors, OUTPUT

METER = "Panther_education_Teofila"
SOURCE = "https://github.com/buds-lab/building-data-genome-project-2/blob/master/data/meters/cleaned/electricity_cleaned.csv"
DIGEST = "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3"


def run(path):
    with path.open("rb") as file:
        if hashlib.file_digest(file, "sha256").hexdigest() != DIGEST:
            raise ValueError(
                "The source checksum differs from the documented BDG2 release."
            )
    frame = (
        pd.read_csv(path, usecols=["timestamp", METER], parse_dates=["timestamp"])
        .set_index("timestamp")
        .asfreq("h")
    )
    y = frame[METER]
    x = pd.DataFrame(
        {
            "hour": y.index.hour,
            "weekday": y.index.dayofweek,
            "month": y.index.month,
            "yesterday": y.shift(24),
            "last_week": y.shift(168),
        },
        index=y.index,
    )
    valid = x.notna().all(axis=1) & y.notna()
    split = pd.Timestamp("2017-01-01")
    train = valid & (y.index < split)
    test = valid & (y.index >= split)
    model = boost().fit(x[train], y[train])
    result = {
        "building": METER,
        "source": SOURCE,
        "reference": "Miller et al. (2020), Building Data Genome 2, Scientific Data, doi:10.1038/s41597-020-00712-x.",
        "sha256": DIGEST,
        "training_hours": int(train.sum()),
        "test_hours": int(test.sum()),
        "missing_meter_hours": int(y.isna().sum()),
        "excluded_hours": int((~valid).sum()),
        "train_year": 2016,
        "test_year": 2017,
        "scores": {
            "Yesterday": errors(y[test], x.loc[test, "yesterday"]),
            "Last week": errors(y[test], x.loc[test, "last_week"]),
            "XGBoost": errors(y[test], np.maximum(0, model.predict(x[test]))),
        },
    }
    OUTPUT.mkdir(exist_ok=True, parents=True)
    (OUTPUT / "real_meter.json").write_text(
        json.dumps(result, indent=2), encoding="utf8"
    )
    print(result["scores"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source_csv", type=Path)
    run(parser.parse_args().source_csv)
