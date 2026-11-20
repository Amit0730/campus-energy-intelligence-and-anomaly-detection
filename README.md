# Campus Energy Consumption Intelligence & Anomaly Detection

A facilities reviewer can compare five simulated university buildings, forecast the next day, investigate abnormal readings, record a decision and export an inspection report. A second experiment benchmarks forecasting on a measured education-building meter.

[Open the Streamlit app](https://campus-energy-intelligence.streamlit.app/) · [Download the assessment report](data/day_ahead/assessment/Campus_Energy_Report.pdf) · [Download the viva slides](data/day_ahead/assessment/Campus_Energy_Viva.pptx)

## Demonstrate the project

1. Open Campus overview and compare actual and forecast electricity with a building or date filter.
2. Choose **Open incident** or open **Project details → Open the overnight case study** for the September 25 Academic Block example.
3. Review the chart, occupancy, expected energy and estimated cost. Enter an inspection note, save a status, refresh the same workspace link and download the inspection report. A workspace link grants access to its review notes. The default SQLite database survives page refreshes. Hosted filesystem replacement can clear local records. Set `ENERGY_REVIEW_DB` to a file on a persistent volume for long-term retention.
4. In Building explorer inspect the weekly heatmap and weekday/weekend profiles. In Forecast lab generate the 24 hours beyond the built-in history and inspect the peak's model contributions, uncertainty and load-shifting scenario.
5. Open the experiments beneath Forecast lab and Incident desk to compare models, inspect feature-removal results and identify actual missed labelled hours in the test set.

## Reproduce

Use Python 3.11 in the project directory.

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app loads the checked-in day-ahead artifacts. For offline training and exports run:

```sh
python -m pip install -r requirements-assessment.txt
python train.py
python -m unittest discover -s tests
```

The dashboard, PDF and slides read the same [experiment results](data/day_ahead/experiment.json), [held-out predictions](data/day_ahead/evaluation.csv.gz) and [independent benchmark](data/day_ahead/real_meter.json). The original one-hour analysis remains available through `historical_dashboard.py` and `train_historical.py`. Its scores refer to a different prediction task.

## Methods and scope

- Five generated campus facilities have hourly electricity, simulated occupancy, weather, equipment channels and injected fault labels. Dates in the monitoring views refer to the historical test period. The first 14 days supply lag history.
- Issue all 24 forecasts at midnight. Use calendar and floor area, yesterday's hourly temperature, last week's occupancy, earlier electricity readings and the previous full day's mean. Train, select, calibrate and test in that order using common calendar boundaries. Compare yesterday, last week, a per-building seasonal autoregression (SARIMA(0,0,0)(1,0,0)[24], fitted by conditional least squares), Ridge and XGBoost.
- The winner minimizes validation RMSE. Compare held-out errors after selection. The nominal 80% hourly prediction range uses a fixed 80th percentile of absolute calibration errors for each facility. Report empirical coverage and feature-removal experiments.
- Isolation Forest fits observed contexts and forecast errors from the selection period. For each detector and building, calibration fault labels tune the standard threshold by F1. Scoring new readings uses observed consumption and occupancy after that hour. Adjacent flags in the same direction are grouped into incidents. Risk is the observed share of flagged hours in the selected period. Explanations suggest inspections. Excess cost is a tariff-based scenario.
- The independent measured-meter experiment trains a separate calendar/lag model in 2016 and tests it in 2017. The meter has no verified fault labels. To reproduce it, obtain the cleaned electricity CSV from [Building Data Genome 2](https://github.com/buds-lab/building-data-genome-project-2/tree/master/data/meters/cleaned) and run `python -m src.real_meter electricity_cleaned.csv`, then `python -m src.assessment`. The script checks the published LFS file hash. Source: Miller et al., [Scientific Data (2020)](https://doi.org/10.1038/s41597-020-00712-x).

Project contributors: Amit Kumar Singh and Atharva Rotake.
