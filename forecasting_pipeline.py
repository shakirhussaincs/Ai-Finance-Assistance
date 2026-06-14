"""
pipelines/forecasting_pipeline.py
Pipeline 3 — Spending Forecasting
  Primary  : Facebook Prophet (per-category monthly forecast)
  Fallback : Linear Regression (for sparse categories)

Uses 33 months of real personal transaction data (Jul 2022 – Mar 2025).
"""
from __future__ import annotations
import os, sys, logging, pickle, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_PROPHET_OK = False
try:
    from prophet import Prophet
    _PROPHET_OK = True
except ImportError:
    logger.warning("Prophet not available – using LinearRegression for all forecasts.")

MIN_PROPHET_MONTHS = 4   # need at least 4 data points for Prophet


def _monthly_agg(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Return monthly spend totals for one category as Prophet-ready df [ds, y]."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    # Only expense transactions (positive amounts in personal dataset)
    mask    = (df["category"] == category) & (df["amount"] > 0)
    monthly = (
        df[mask]
        .set_index("date")["amount"]
        .resample("MS")
        .sum()
        .reset_index()
        .rename(columns={"date": "ds", "amount": "y"})
    )
    monthly["ds"] = monthly["ds"].dt.tz_localize(None)  # Prophet needs tz-naive
    return monthly[monthly["y"] > 0]  # skip months with zero spend


#Linear Regression fallback

class _LinearForecaster:
    def __init__(self):
        self.model   = LinearRegression()
        self.scaler  = StandardScaler()
        self.n_obs   = 0
        self.last_ds = None

    def fit(self, monthly: pd.DataFrame):
        X = np.arange(len(monthly)).reshape(-1, 1)
        y = monthly["y"].values
        self.model.fit(self.scaler.fit_transform(X), y)
        self.n_obs   = len(monthly)
        self.last_ds = monthly["ds"].iloc[-1]

    def predict(self, months_ahead: int) -> pd.DataFrame:
        future_idx = np.arange(self.n_obs, self.n_obs + months_ahead).reshape(-1, 1)
        yhat = self.model.predict(self.scaler.transform(future_idx))
        yhat = np.maximum(yhat, 0)
        future_dates = pd.date_range(
            start=self.last_ds + pd.DateOffset(months=1),
            periods=months_ahead, freq="MS"
        )
        return pd.DataFrame({
            "ds": future_dates, "yhat": yhat,
            "yhat_lower": yhat * 0.85,
            "yhat_upper": yhat * 1.15,
        })


#Prophet wrapper 

class _ProphetForecaster:
    def __init__(self):
        self.model = None

    def fit(self, monthly: pd.DataFrame):
        self.model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            interval_width=0.80,
            changepoint_prior_scale=0.05,
        )
        self.model.fit(monthly[["ds", "y"]])

    def predict(self, months_ahead: int) -> pd.DataFrame:
        future   = self.model.make_future_dataframe(periods=months_ahead, freq="MS")
        forecast = self.model.predict(future)
        result   = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(months_ahead)
        result["yhat"]       = result["yhat"].clip(lower=0)
        result["yhat_lower"] = result["yhat_lower"].clip(lower=0)
        return result.reset_index(drop=True)


#Main Pipeline 

class ForecastingPipeline:
    """One forecaster per spending category."""

    def __init__(self):
        self._models:      Dict[str, Any]  = {}
        self._model_types: Dict[str, str]  = {}
        self._monthly_data: Dict[str, pd.DataFrame] = {}

    def train(self, df: pd.DataFrame, categories: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Train on combined transaction DataFrame.
        Returns {category: model_type_used}.
        """
        if categories is None:
            categories = [c for c in df["category"].unique()
                          if c not in ("Income", "Transfer", "Refund")]

        summary = {}
        for cat in categories:
            monthly = _monthly_agg(df, cat)
            self._monthly_data[cat] = monthly

            if len(monthly) < 2:
                logger.info(f"[Forecast] '{cat}': only {len(monthly)} month(s) — skipping")
                continue

            if _PROPHET_OK and len(monthly) >= MIN_PROPHET_MONTHS:
                try:
                    m = _ProphetForecaster()
                    m.fit(monthly)
                    self._models[cat]      = m
                    self._model_types[cat] = "prophet"
                    summary[cat]           = "prophet"
                    logger.info(f"[Forecast] '{cat}': Prophet trained on {len(monthly)} months")
                    continue
                except Exception as e:
                    logger.warning(f"[Forecast] '{cat}': Prophet failed ({e}), falling back")

            m = _LinearForecaster()
            m.fit(monthly)
            self._models[cat]      = m
            self._model_types[cat] = "linear"
            summary[cat]           = "linear"
            logger.info(f"[Forecast] '{cat}': LinearRegression ({len(monthly)} months)")

        return summary

    def forecast(self, months_ahead: int = 3,
                 categories: Optional[List[str]] = None) -> pd.DataFrame:
        """Return forecasts for all (or selected) categories."""
        cats = categories or list(self._models.keys())
        frames = []
        for cat in cats:
            if cat not in self._models:
                continue
            df_pred = self._models[cat].predict(months_ahead)
            df_pred["category"]  = cat
            df_pred["model_used"] = self._model_types[cat]
            frames.append(df_pred)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def forecast_total(self, months_ahead: int = 3) -> pd.DataFrame:
        """Sum all category forecasts into total monthly spending."""
        df = self.forecast(months_ahead)
        if df.empty:
            return df
        total = df.groupby("ds")[["yhat","yhat_lower","yhat_upper"]].sum().reset_index()
        total["category"]   = "TOTAL"
        total["model_used"] = "aggregated"
        return total

    def historical_monthly(self) -> pd.DataFrame:
        """Return actual monthly data used for training (for charting)."""
        frames = []
        for cat, monthly in self._monthly_data.items():
            m = monthly.copy()
            m["category"] = cat
            frames.append(m)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def save(self, directory: str | Path):
        import json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for cat, model in self._models.items():
            safe = cat.lower().replace(" ", "_").replace("&", "and")
            with open(directory / f"{safe}_{self._model_types[cat]}.pkl", "wb") as f:
                pickle.dump(model, f)
        with open(directory / "index.json", "w") as f:
            json.dump(self._model_types, f, indent=2)
        logger.info(f"[Forecast] Saved {len(self._models)} models → {directory}")

    def load(self, directory: str | Path):
        import json
        directory = Path(directory)
        with open(directory / "index.json") as f:
            self._model_types = json.load(f)
        for cat, mtype in self._model_types.items():
            safe = cat.lower().replace(" ", "_").replace("&", "and")
            p = directory / f"{safe}_{mtype}.pkl"
            if p.exists():
                with open(p, "rb") as f:
                    self._models[cat] = pickle.load(f)
        logger.info(f"[Forecast] Loaded {len(self._models)} models ← {directory}")


# Self-test 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data_prep import build_combined

    data = build_combined(
        "dataset/11 march 2025.csv",
        "dataset/banking-transaction-categorization-dataset.csv",
    )
    df = data["combined"]

    pipe = ForecastingPipeline()
    summary = pipe.train(df)

    print(f"\nModels trained: {len(summary)}")
    for cat, mtype in summary.items():
        print(f"  {cat:<22} → {mtype}")

    # 3-month forecast
    fc = pipe.forecast(months_ahead=3)
    total = pipe.forecast_total(months_ahead=3)

    print(f"\n--- 3-Month Category Forecasts ---")
    for _, row in fc.iterrows():
        print(f"  {row['category']:<22} {str(row['ds'])[:7]}  ${row['yhat']:.0f}  "
              f"[${row['yhat_lower']:.0f}–${row['yhat_upper']:.0f}] ({row['model_used']})")

    print(f"\n--- Monthly Total Forecast ---")
    for _, row in total.iterrows():
        print(f"  {str(row['ds'])[:7]}  ${row['yhat']:.0f}  [${row['yhat_lower']:.0f}–${row['yhat_upper']:.0f}]")

    pipe.save("/home/claude/finance_coach/models/forecast_models")
    print("\nModels saved.")
