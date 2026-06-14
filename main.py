"""
api/main.py  —  AI Personal Finance Coach REST API
Loads all 3 trained models on startup and exposes clean endpoints.
"""
from __future__ import annotations
import logging, os, sys, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_prep import build_combined, monthly_summary, CANONICAL_CATEGORIES
from categorization_pipeline import CategorizationPipeline
from anomaly_pipeline import AnomalyPipeline
from forecasting_pipeline import ForecastingPipeline
from llm_coach import LLMCoach

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH_PERSONAL = os.getenv("DATA_PERSONAL", "dataset/11 march 2025.csv")
DATA_PATH_BANKING  = os.getenv("DATA_BANKING",  "dataset/banking-transaction-categorization-dataset.csv")
MODEL_DIR          = os.getenv("MODEL_DIR",      "./models")
GEMINI_KEY      = os.getenv("GEMINI_API_KEY", "your_key_here")


# ── App State ─────────────────────────────────────────────────────────────────
class State:
    categorizer:  CategorizationPipeline | None = None
    anomaly:      AnomalyPipeline         | None = None
    forecaster:   ForecastingPipeline     | None = None
    coach:        LLMCoach                | None = None
    df_combined:  pd.DataFrame            | None = None
    last_trained: str                           = "never"

state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Starting AI Finance Coach API ===")
    _load_and_train()
    yield
    logger.info("=== Shutting down ===")


def _load_and_train():
    """Load datasets, train all pipelines, initialise coach."""
    try:
        data = build_combined(DATA_PATH_PERSONAL, DATA_PATH_BANKING)
        state.df_combined = data["combined"]

        # 1. Categorizer
        cat_path = f"{MODEL_DIR}/tfidf_categorizer.joblib"
        state.categorizer = CategorizationPipeline()
        if os.path.exists(cat_path):
            state.categorizer.load(cat_path)
        else:
            state.categorizer.train(data["categorization_train"])
            os.makedirs(MODEL_DIR, exist_ok=True)
            state.categorizer.save(cat_path)

        # 2. Anomaly
        anom_path = f"{MODEL_DIR}/anomaly_model.joblib"
        state.anomaly = AnomalyPipeline()
        if os.path.exists(anom_path):
            state.anomaly.load(anom_path)
        else:
            state.anomaly.train(state.df_combined)
            state.anomaly.save(anom_path)

        # 3. Forecaster
        fc_dir = f"{MODEL_DIR}/forecast_models"
        state.forecaster = ForecastingPipeline()
        if os.path.exists(f"{fc_dir}/index.json"):
            state.forecaster.load(fc_dir)
        else:
            state.forecaster.train(state.df_combined)
            state.forecaster.save(fc_dir)

        # 4. LLM Coach
        state.coach = LLMCoach(api_key=GEMINI_KEY)

        state.last_trained = datetime.utcnow().isoformat()
        logger.info("All pipelines ready.")
    except Exception as e:
        logger.error(f"Startup error: {e}", exc_info=True)


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Personal Finance Coach",
    description="ML pipelines: Categorization · Anomaly Detection · Forecasting · Gemini LLM",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Schemas ────────────────────────────────────────────────────────────────────
class TransactionIn(BaseModel):
    description: str
    amount: float
    date: Optional[str] = None
    category: Optional[str] = None
    is_recurring: bool = False

class CategorizeReq(BaseModel):
    descriptions: List[str]

class AnomalyReq(BaseModel):
    transactions: List[TransactionIn]

class ForecastReq(BaseModel):
    months_ahead: int = Field(3, ge=1, le=12)
    categories: Optional[List[str]] = None

class ChatReq(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = None
    user_profile: Optional[Dict[str, Any]] = None


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "categorizer":  state.categorizer is not None and state.categorizer._fitted,
            "anomaly":      state.anomaly is not None and state.anomaly._fitted,
            "forecaster":   state.forecaster is not None and bool(state.forecaster._models),
            "llm_coach":    state.coach is not None,
            "gemini_active": state.coach.using_gemini if state.coach else False,
        },
        "dataset_rows": len(state.df_combined) if state.df_combined is not None else 0,
        "last_trained": state.last_trained,
    }


@app.get("/api/v1/summary")
def get_summary():
    """Dashboard summary: category totals, monthly trend, top stats."""
    if state.df_combined is None:
        raise HTTPException(503, "Data not loaded")
    df = state.df_combined
    # Overall spend by category
    cat_totals = (
        df[df["amount"] > 0]
        .groupby("category")["amount"].sum()
        .sort_values(ascending=False)
        .round(2).to_dict()
    )
    # Last 6 months trend
    df["month"] = df["date"].dt.to_period("M")
    monthly = (
        df[df["amount"] > 0]
        .groupby("month")["amount"].sum()
        .tail(6)
        .reset_index()
    )
    monthly["month"] = monthly["month"].astype(str)
    return {
        "category_totals":    cat_totals,
        "monthly_trend":      monthly.to_dict(orient="records"),
        "total_transactions": len(df),
        "date_range":         {
            "start": str(df["date"].min())[:10],
            "end":   str(df["date"].max())[:10],
        },
        "categories": CANONICAL_CATEGORIES,
    }


@app.post("/api/v1/categorize")
def categorize(req: CategorizeReq):
    """Classify transaction descriptions into spending categories."""
    if not state.categorizer or not state.categorizer._fitted:
        raise HTTPException(503, "Categorizer not ready")
    results = state.categorizer.predict(req.descriptions)
    return {"results": results, "model": "tfidf_lr", "n": len(results)}


@app.post("/api/v1/anomalies")
def detect_anomalies(req: AnomalyReq):
    """Detect spending anomalies in a list of transactions."""
    if not state.anomaly or not state.anomaly._fitted:
        raise HTTPException(503, "Anomaly model not ready")
    df = pd.DataFrame([t.model_dump() for t in req.transactions])
    df["date"] = pd.to_datetime(df.get("date", datetime.utcnow().isoformat()), utc=True, errors="coerce")
    result = state.anomaly.detect(df)
    anomalies = result[result["severity"] != "NORMAL"]
    return {
        "total":            len(result),
        "anomalies_found":  len(anomalies),
        "severity_counts":  result["severity"].value_counts().to_dict(),
        "anomalies":        anomalies[["description","amount","category","severity",
                                       "z_score","iso_score","anomaly_reason","date"]]
                            .to_dict(orient="records"),
    }


@app.post("/api/v1/forecast")
def forecast(req: ForecastReq):
    """Forecast future monthly spending per category using Prophet."""
    if not state.forecaster or not state.forecaster._models:
        raise HTTPException(503, "Forecaster not ready")
    fc = state.forecaster.forecast(
        months_ahead=req.months_ahead,
        categories=req.categories,
    )
    total = state.forecaster.forecast_total(months_ahead=req.months_ahead)
    fc["ds"]    = fc["ds"].astype(str)
    total["ds"] = total["ds"].astype(str)
    return {
        "months_ahead":  req.months_ahead,
        "by_category":   fc.to_dict(orient="records"),
        "total":         total.to_dict(orient="records"),
    }


@app.post("/api/v1/chat")
def chat(req: ChatReq):
    """Multi-turn financial coaching chat powered by Gemini."""
    if not state.coach:
        raise HTTPException(503, "Coach not ready")
    return state.coach.chat(
        user_message=req.message,
        context=req.context,
        history=req.history,
        user_profile=req.user_profile,
    )


@app.post("/api/v1/monthly-report")
def monthly_report(req: ChatReq):
    """Generate a proactive monthly financial health report via Gemini."""
    if not state.coach:
        raise HTTPException(503, "Coach not ready")
    return state.coach.monthly_report(
        context=req.context or {},
        user_profile=req.user_profile,
    )


@app.post("/api/v1/analyze")
def full_analyze(transactions: List[TransactionIn]):
    """
    Full pipeline on a new batch of transactions:
    1. Categorize → 2. Detect anomalies → 3. Return ML context ready for chat.
    """
    if not state.categorizer or not state.anomaly:
        raise HTTPException(503, "Models not ready")

    df = pd.DataFrame([t.model_dump() for t in transactions])
    df["date"] = pd.to_datetime(df.get("date", datetime.utcnow().isoformat()), utc=True, errors="coerce")

    # Step 1: categorize
    cat_results = state.categorizer.predict(df["description"].tolist())
    for i, r in enumerate(cat_results):
        if pd.isna(df.loc[i, "category"]) or df.loc[i, "category"] is None:
            df.loc[i, "category"] = r["category"]

    # Step 2: anomalies
    anomaly_result = state.anomaly.detect(df)
    anomalies = anomaly_result[anomaly_result["severity"] != "NORMAL"]

    # Step 3: spending summary
    spending = (
        df[df["amount"] > 0]
        .groupby("category")["amount"].sum()
        .sort_values(ascending=False)
        .round(2).to_dict()
    )

    return {
        "transaction_count":  len(df),
        "categorization":     cat_results,
        "spending_summary":   spending,
        "anomalies_found":    len(anomalies),
        "anomalies":          anomalies[["description","amount","category","severity","z_score"]].to_dict(orient="records"),
        "ml_context": {
            "spending_summary": spending,
            "anomalies": anomalies[["amount","category","severity","z_score"]].to_dict(orient="records"),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
