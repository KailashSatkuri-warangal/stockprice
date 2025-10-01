# Stock Price Prediction — README

**Project:** Stock Price Prediction (Kaggle dataset)

**Author:** Satkuri Kailash

---

## Project Overview
This repository contains code and notebooks for training, evaluating, and serving time-series models that predict stock prices (close price / next-day return) using a Kaggle dataset. The goal of this README is to give students and contributors a clear, reproducible path from raw dataset (Kaggle) to a working Streamlit demo.

## Live Demo
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://reviewvibe.streamlit.app/)

Open the link above to see the running Streamlit app for this project.

---

## Dataset
**Source:** Kaggle — *stock price* dataset (replace with the exact Kaggle dataset slug you used).

**How to obtain (recommended):**
1. Install Kaggle CLI if you don't have it: `pip install kaggle`
2. Configure credentials: place `kaggle.json` under `~/.kaggle/kaggle.json` (or follow Kaggle docs).
3. Download dataset (example):

```bash
kaggle datasets download -d <owner/dataset-slug> -p data/ --unzip
```

**Common file(s) included:**
- `data/stock_prices.csv` — daily OHLCV (Open, High, Low, Close, Volume) with `date`, `ticker` (optional), and other fields.

**Typical columns** (may vary by dataset):
- `Date` or `date` — trading day (YYYY-MM-DD)
- `Open` — opening price
- `High` — highest price
- `Low` — lowest price
- `Close` — closing price
- `Adj Close` — adjusted close (if provided)
- `Volume` — traded volume
- `Ticker` — symbol (if dataset contains multiple stocks)

> **Teacher's tip:** keep a copy of the raw CSV in `data/raw/` and *never* overwrite it — keep reproducibility.

---

## Repository Structure (recommended)
```
stock-predict/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/ datasets -0,1,2,3
│  ├─ processed/
├─ notebooks/
│  ├─ 01-exploration
│  ├─ 02-preprocessing
│  ├─ 03-training
├─ src/
│  ├─ data_loader.py
│  ├─ preprocess.py
│  ├─ features.py
│  ├─ models.py
│  ├─ train.py
│  ├─ predict.py
├─ models/
├─ app/
│  ├─ app.py
└─ tests/
```

---

## Setup (local)
1. Create a Python virtual environment and activate it (use your OS preferred method).

```bash
python -m venv venv
# windows
venv\Scripts\activate
# linux / mac
source venv/bin/activate
pip install -r requirements.txt
```

2. `requirements.txt` (example):
```
pandas
numpy
scikit-learn
matplotlib
pandas-datareader
streamlit
joblib
xgboost
tensorflow # optional
kaggle
```

> **Lab note:** this project was developed for reproducibility; pin package versions in an actual assignment.

---

## Preprocessing
1. **Load raw CSV** from `data/raw/`.
2. **Sort by date** and set `date` as pandas `DatetimeIndex`.
3. **Fill missing days** (optional) — forward-fill only where appropriate; do not invent prices for long gaps.
4. **Feature engineering suggestions:**
   - Returns: `close_pct_change = close.pct_change()`
   - Moving averages: `ma_5`, `ma_10`, `ma_20`
   - Volatility: rolling std of returns
   - Volume features: `vol_ma` or volume changes
   - Lag features: `close_lag_1`, `close_lag_2`
   - Technical indicators (RSI, MACD) — optional for advanced students

5. **Train / Val / Test split** — use time-based split (no random shuffles). Example: 70% train, 15% val, 15% test by date.

> **Common pitfall:** do not use future data to compute features at training time. For example compute rolling features using only historical window.

---

## Models
This repo supports multiple model choices — baseline and advanced:
- **Baseline:** Linear Regression on lag and moving-average features.
- **Tree-based:** XGBoost / RandomForest for non-linear patterns.
- **Neural:** LSTM or 1D-CNN for sequence forecasting (advanced).

**Training command (example):**
```bash
python src/train.py --config config/train_config.yaml
```

`train.py` should:
- load processed data
- create datasets and dataloaders
- fit model and save best checkpoint to `models/`
- log metrics to console and an experiment CSV

---

## Evaluation
Metrics to report (choose per problem formulation):
- **Regression:** RMSE, MAE, MAPE
- **Direction accuracy:** percent of times predicted direction (up/down) matches actual

Plot predictions vs ground truth on test set and include residual analysis.

---

## Inference / Predict
Run prediction on a single day or a range using `src/predict.py`.

```bash
python src/predict.py --model models/best_model.pkl --start 2024-01-01 --end 2024-02-01
```

The Streamlit app in `app/streamlit_app.py` calls the same `predict` utilities to display live charts.

---

## Quick Run (development)
1. Download dataset into `data/raw/`.
2. Preprocess and create processed CSV(s):
```bash
python src/preprocess.py --input data/raw/stock_prices.csv --output data/processed/processed.csv
```
3. Train quick baseline model:
```bash
python src/train.py --quick True
```
4. Launch Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

---

## Reproducibility & Notes
- Keep `random_state` fixed in all model training and splitting utilities.
- Save processing pipeline (scaler, encoders) with the model (e.g., `joblib.dump`).
- Use time-based cross-validation for hyperparameter search.

---

## Tests
Add small unit tests in `tests/` to validate:
- `data_loader` reads expected columns
- `preprocess` output shapes and date ordering
- `predict` function returns values in expected ranges

---

## Common errors & fixes (lab instructor notes)
- `Error: 'DataFrame' object has no attribute 'tolist'` — check if you're calling `.tolist()` on a DataFrame instead of a Series; use `df['col'].tolist()`.
- `sqlite3.OperationalError: no such column: timestamp` — verify your SQL query and table schema when using databases.
- If your Streamlit app shows wrong data, confirm the app points to `data/processed/` not `data/raw/`.

---

## License
Choose an appropriate license (e.g., MIT) and add `LICENSE` to the repo.

---

## Contact
For questions or to propose changes, open an issue or contact the author (Satkuri Kailash).

---

*End of README.*

