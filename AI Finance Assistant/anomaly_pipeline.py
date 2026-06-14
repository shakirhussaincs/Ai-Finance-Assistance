"""
pipelines/anomaly_pipeline.py
Pipeline 2 — Anomaly Detection
  Primary  : Isolation Forest  (unsupervised, no labels needed)
  Fallback : Z-Score threshold

Trained on 4797 real transactions from the merged dataset.
"""
from __future__ import annotations
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

logger = logging.getLogger(__name__)


# Feature scaling

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build numerical feature matrix from real transaction data.
    Works with columns: date, amount, category, is_recurring
    """
    df = df.copy()
    df["date"]   = pd.to_datetime(df["date"], utc=True, errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs().fillna(0)
    df = df.sort_values("date").reset_index(drop=True)

   
    df["rolling_mean_7d"] = df["amount"].rolling(window=7, min_periods=1).mean()
    df["rolling_std_7d"]  = df["amount"].rolling(window=7, min_periods=1).std().fillna(0)
    df["amount_vs_mean"]  = df["amount"] - df["rolling_mean_7d"]

    cat_stats = df.groupby("category")["amount"].agg(["mean", "std"]).rename(
        columns={"mean": "cat_mean", "std": "cat_std"}
    )
    cat_stats["cat_std"] = cat_stats["cat_std"].fillna(1.0)
    df = df.join(cat_stats, on="category")
    df["amount_vs_cat_mean"] = df["amount"] - df["cat_mean"]

   
    df["hour"]      = df["date"].dt.hour
    df["dow"]       = df["date"].dt.dayofweek
    df["month"]     = df["date"].dt.month
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"]   / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"]   / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

   
    df["is_recurring_int"] = df.get("is_recurring", pd.Series([False]*len(df))).astype(int)

    feature_cols = [
        "amount", "rolling_mean_7d", "rolling_std_7d",
        "amount_vs_mean", "amount_vs_cat_mean",
        "cat_mean", "cat_std",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_recurring_int",
    ]
    return df[feature_cols].fillna(0)




class AnomalyPipeline:
    """
    Combines Isolation Forest (primary) and Z-Score (fallback).
    Severity: HIGH = both flag, MEDIUM = only IF, LOW = only Z, NORMAL = neither.
    """

    SEVERITY_MATRIX = {
        (True,  True):  "HIGH",
        (True,  False): "MEDIUM",
        (False, True):  "LOW",
        (False, False): "NORMAL",
    }

    def __init__(self, contamination: float = 0.05, z_threshold: float = 3.0):
        self.contamination = contamination
        self.z_threshold   = z_threshold
        self.scaler        = StandardScaler()
        self.iso_forest    = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=42,
        )
        self._fitted   = False
        self._z_mean: Optional[float] = None
        self._z_std:  Optional[float] = None

   

    def train(self, df: pd.DataFrame) -> Dict[str, Any]:
        features  = build_features(df)
        X_scaled  = self.scaler.fit_transform(features.values)
        self.iso_forest.fit(X_scaled)

        amounts        = df["amount"].abs().fillna(0).values
        self._z_mean   = float(np.mean(amounts))
        self._z_std    = float(np.std(amounts)) or 1.0
        self._fitted   = True

        
        labels = self.iso_forest.predict(X_scaled)
        n_anomalies = int((labels == -1).sum())
        logger.info(f"[Anomaly] Trained on {len(df)} txns | IF flagged {n_anomalies} ({n_anomalies/len(df)*100:.1f}%)")
        return {
            "n_train":    len(df),
            "n_flagged":  n_anomalies,
            "pct_flagged": round(n_anomalies / len(df) * 100, 2),
            "z_mean":     round(self._z_mean, 2),
            "z_std":      round(self._z_std, 2),
        }



    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score transactions. Returns input df with extra columns:
          iso_score, z_score, is_iso_anomaly, is_z_anomaly, severity, anomaly_reason
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() first.")

        features  = build_features(df)
        X_scaled  = self.scaler.transform(features.values)

        iso_labels = self.iso_forest.predict(X_scaled)       
        iso_scores = self.iso_forest.score_samples(X_scaled)
        amounts    = df["amount"].abs().fillna(0).values
        z_scores   = np.abs((amounts - self._z_mean) / self._z_std)
        z_labels   = np.where(z_scores > self.z_threshold, -1, 1)

        result = df.copy().reset_index(drop=True)
        result["iso_score"]      = iso_scores.round(4)
        result["z_score"]        = z_scores.round(3)
        result["is_iso_anomaly"] = iso_labels == -1
        result["is_z_anomaly"]   = z_labels   == -1
        result["severity"]       = [
            self.SEVERITY_MATRIX[(bool(iso_labels[i]==-1), bool(z_labels[i]==-1))]
            for i in range(len(result))
        ]
        result["anomaly_reason"] = [
            self._reason(result.iloc[i], iso_labels[i], z_scores[i])
            for i in range(len(result))
        ]
        return result

    @staticmethod
    def _reason(row, iso_label, z_score) -> str:
        parts = []
        if iso_label == -1:
            parts.append("Isolation Forest: unusual pattern in multi-dimensional spending space")
        if z_score > 3.0:
            parts.append(f"Z-Score: {z_score:.1f}σ above personal spending mean")
        return ". ".join(parts) if parts else "Within normal range"


    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler, "iso_forest": self.iso_forest,
            "z_mean": self._z_mean, "z_std": self._z_std,
        }, path)
        logger.info(f"[Anomaly] Saved → {path}")

    def load(self, path: str | Path):
        data = joblib.load(path)
        self.scaler     = data["scaler"]
        self.iso_forest = data["iso_forest"]
        self._z_mean    = data["z_mean"]
        self._z_std     = data["z_std"]
        self._fitted    = True
        logger.info(f"[Anomaly] Loaded ← {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    from data_prep import build_combined

    data = build_combined(
        "dataset/11 march 2025.csv",
        "dataset/banking-transaction-categorization-dataset.csv",
    )
    df = data["combined"]

    pipe = AnomalyPipeline(contamination=0.05)
    stats = pipe.train(df)
    print(f"\nTraining stats: {stats}")

    # Run detection
    result = pipe.detect(df)
    anomalies = result[result["severity"] != "NORMAL"]

    print(f"\nTotal anomalies: {len(anomalies)} / {len(df)}")
    print(f"\nSeverity breakdown:\n{result['severity'].value_counts().to_string()}")

    print(f"\n--- Top 10 HIGH/MEDIUM anomalies ---")
    top = anomalies[anomalies["severity"].isin(["HIGH","MEDIUM"])].nlargest(10, "amount")
    for _, row in top.iterrows():
        print(f"  [{row['severity']:<6}] ${row['amount']:<8.2f} | {row['category']:<20} | z={row['z_score']:.1f}σ | {str(row['date'])[:10]}")

    pipe.save("/home/claude/finance_coach/models/anomaly_model.joblib")
    print("\nModel saved.")
