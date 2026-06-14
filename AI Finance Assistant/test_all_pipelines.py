"""
tests/test_all_pipelines.py
Full test suite — uses real uploaded datasets.
Run: pytest tests/ -v
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pandas as pd
import numpy as np

PERSONAL_CSV = "dataset/11 march 2025.csv"
BANKING_CSV  = "dataset/banking-transaction-categorization-dataset.csv"


# Fixtures

@pytest.fixture(scope="session")
def real_data():
    from data_prep import build_combined
    return build_combined(PERSONAL_CSV, BANKING_CSV)

@pytest.fixture(scope="session")
def trained_categorizer(real_data):
    from categorization_pipeline import CategorizationPipeline
    pipe = CategorizationPipeline()
    pipe.train(real_data["categorization_train"])
    return pipe

@pytest.fixture(scope="session")
def trained_anomaly(real_data):
    from anomaly_pipeline import AnomalyPipeline
    pipe = AnomalyPipeline(contamination=0.05)
    pipe.train(real_data["combined"])
    return pipe

@pytest.fixture(scope="session")
def trained_forecaster(real_data):
    from forecasting_pipeline import ForecastingPipeline
    pipe = ForecastingPipeline()
    pipe.train(real_data["combined"])
    return pipe


#1. Data Prep Tests 

class TestDataPrep:
    def test_personal_loads(self, real_data):
        df = real_data["personal"]
        assert len(df) == 4597
        assert "category" in df.columns
        assert "amount" in df.columns
        assert "date" in df.columns

    def test_banking_loads(self, real_data):
        df = real_data["banking"]
        assert len(df) == 200
        assert "merchant" in df.columns

    def test_combined_size(self, real_data):
        assert len(real_data["combined"]) == 4797

    def test_no_null_amounts(self, real_data):
        assert real_data["combined"]["amount"].isnull().sum() == 0

    def test_canonical_categories(self, real_data):
        from data_prep import CANONICAL_CATEGORIES
        cats = real_data["combined"]["category"].unique()
        for cat in cats:
            assert cat in CANONICAL_CATEGORIES, f"Unknown category: {cat}"

    def test_date_range(self, real_data):
        df = real_data["personal"]
        assert df["date"].min().year == 2022
        assert df["date"].max().year == 2025

    def test_train_set_labelled(self, real_data):
        train = real_data["categorization_train"]
        assert len(train) > 4000
        assert "description" in train.columns
        assert "category" in train.columns
        assert train["category"].isnull().sum() == 0


#2. Categorization Pipeline Tests 

class TestCategorization:
    def test_accuracy_above_95(self, trained_categorizer, real_data):
        """Model must achieve >95% accuracy on real data."""
        from sklearn.model_selection import cross_val_score
        train = real_data["categorization_train"]
        texts  = train["description"].str.lower().tolist()
        labels = trained_categorizer.label_encoder.transform(train["category"].tolist())
        scores = cross_val_score(trained_categorizer.pipeline, texts, labels, cv=3)
        assert scores.mean() > 0.95, f"Accuracy too low: {scores.mean():.4f}"

    def test_predict_restaurant(self, trained_categorizer):
        result = trained_categorizer.predict_single("Restuarant")
        assert result["category"] == "Food & Dining"
        assert result["confidence"] > 0.90

    def test_predict_coffee(self, trained_categorizer):
        result = trained_categorizer.predict_single("Coffe")
        assert result["category"] == "Coffee & Cafes"

    def test_predict_market(self, trained_categorizer):
        result = trained_categorizer.predict_single("Market")
        assert result["category"] == "Groceries"

    def test_predict_taxi(self, trained_categorizer):
        result = trained_categorizer.predict_single("Taxi")
        assert result["category"] == "Transport"

    def test_predict_uber(self, trained_categorizer):
        result = trained_categorizer.predict_single("Uber trip CA3245")
        assert result["category"] == "Transport"

    def test_predict_netflix(self, trained_categorizer):
        result = trained_categorizer.predict_single("Netflix subscription")
        assert result["category"] == "Entertainment"

    def test_predict_returns_top3(self, trained_categorizer):
        result = trained_categorizer.predict_single("Starbucks coffee")
        assert "top3" in result
        assert len(result["top3"]) == 3
        assert all("probability" in t for t in result["top3"])

    def test_batch_predict(self, trained_categorizer):
        descs = ["Market", "Taxi", "Coffe", "Restuarant", "Sport"]
        results = trained_categorizer.predict(descs)
        assert len(results) == 5
        assert all("confidence" in r for r in results)

    def test_save_load_roundtrip(self, trained_categorizer, tmp_path):
        from categorization_pipeline import CategorizationPipeline
        p = tmp_path / "cat.joblib"
        trained_categorizer.save(p)
        pipe2 = CategorizationPipeline()
        pipe2.load(p)
        r = pipe2.predict_single("Taxi")
        assert r["category"] == "Transport"

    def test_untrained_raises(self):
        from categorization_pipeline import CategorizationPipeline
        with pytest.raises(RuntimeError):
            CategorizationPipeline().predict(["test"])


# 3. Anomaly Detection Tests 

class TestAnomalyDetection:
    def test_flags_5pct(self, trained_anomaly, real_data):
        result = trained_anomaly.detect(real_data["combined"])
        pct = (result["severity"] != "NORMAL").mean()
        assert 0.03 < pct < 0.10, f"Expected ~5% anomalies, got {pct:.2%}"

    def test_banking_spike_detected(self, trained_anomaly, real_data):
        """$1M income entry must be flagged as HIGH."""
        result = trained_anomaly.detect(real_data["banking"])
        high = result[result["severity"] == "HIGH"]
        assert len(high) > 0

    def test_severity_values(self, trained_anomaly, real_data):
        result = trained_anomaly.detect(real_data["combined"].head(100))
        assert set(result["severity"].unique()).issubset({"NORMAL","LOW","MEDIUM","HIGH"})

    def test_extreme_amount_flagged(self, trained_anomaly, real_data):
        df = real_data["combined"].head(50).copy()
        df.loc[0, "amount"] = 99999.0
        result = trained_anomaly.detect(df)
        assert result.loc[0, "severity"] in ("HIGH", "MEDIUM")

    def test_has_z_score_column(self, trained_anomaly, real_data):
        result = trained_anomaly.detect(real_data["combined"].head(20))
        assert "z_score" in result.columns
        assert "iso_score" in result.columns

    def test_save_load(self, trained_anomaly, real_data, tmp_path):
        from anomaly_pipeline import AnomalyPipeline
        p = tmp_path / "anomaly.joblib"
        trained_anomaly.save(p)
        pipe2 = AnomalyPipeline()
        pipe2.load(p)
        result = pipe2.detect(real_data["combined"].head(10))
        assert len(result) == 10

    def test_detect_before_train_raises(self, real_data):
        from anomaly_pipeline import AnomalyPipeline
        with pytest.raises(RuntimeError):
            AnomalyPipeline().detect(real_data["combined"].head(5))


#4. Forecasting Pipeline Tests 

class TestForecasting:
    def test_trains_all_categories(self, trained_forecaster):
        assert len(trained_forecaster._models) >= 10

    def test_prophet_used_for_main_cats(self, trained_forecaster):
        for cat in ["Food & Dining", "Coffee & Cafes", "Groceries", "Transport"]:
            assert trained_forecaster._model_types.get(cat) == "prophet", \
                f"{cat} should use Prophet (33 months available)"

    def test_forecast_shape(self, trained_forecaster):
        fc = trained_forecaster.forecast(months_ahead=3)
        assert not fc.empty
        assert "yhat" in fc.columns
        assert "yhat_lower" in fc.columns
        assert "yhat_upper" in fc.columns

    def test_yhat_non_negative(self, trained_forecaster):
        fc = trained_forecaster.forecast(months_ahead=3)
        assert (fc["yhat"] >= 0).all()

    def test_forecast_total(self, trained_forecaster):
        total = trained_forecaster.forecast_total(months_ahead=3)
        assert not total.empty
        assert (total["category"] == "TOTAL").all()
        assert len(total) >= 3   # >= 3 because sparse cats may have different base dates

    def test_forecast_months_ahead_param(self, trained_forecaster):
        fc1 = trained_forecaster.forecast(months_ahead=1)
        fc6 = trained_forecaster.forecast(months_ahead=6)
        cats = trained_forecaster._models.keys()
        assert len(fc1) == len(list(cats))
        assert len(fc6) == len(list(cats)) * 6

    def test_save_load(self, trained_forecaster, tmp_path):
        from forecasting_pipeline import ForecastingPipeline
        trained_forecaster.save(tmp_path)
        pipe2 = ForecastingPipeline()
        pipe2.load(tmp_path)
        fc = pipe2.forecast(months_ahead=1)
        assert not fc.empty


#5. Integration Test 

class TestIntegration:
    def test_full_pipeline_on_new_transactions(
        self, trained_categorizer, trained_anomaly, real_data
    ):
        """Simulate new transactions flowing through categorize → anomaly."""
        new_txns = pd.DataFrame({
            "description": ["Restuarant", "Coffe", "Market", "Taxi", "Sport"],
            "amount":       [45.0, 8.5, 120.0, 15.0, 60.0],
            "category":     [None] * 5,
            "date":         pd.to_datetime(["2025-03-01"]*5, utc=True),
            "is_recurring": [False]*5,
        })

        # Step 1: Categorize
        cat_results = trained_categorizer.predict(new_txns["description"].tolist())
        for i, r in enumerate(cat_results):
            new_txns.loc[i, "category"] = r["category"]

        assert new_txns["category"].isnull().sum() == 0

        # Step 2: Anomaly detect
        result = trained_anomaly.detect(new_txns)
        assert "severity" in result.columns
        assert len(result) == 5

        # Step 3: Spending summary
        spending = new_txns.groupby("category")["amount"].sum().to_dict()
        assert len(spending) > 0

    def test_data_flows_to_coach(self, real_data):
        """Verify ML context dict is well-formed for LLM coach."""
        from llm_coach import LLMCoach, _build_context_block
        df = real_data["combined"]
        spending = df.groupby("category")["amount"].sum().sort_values(ascending=False).head(5).to_dict()
        context = {
            "spending_summary": spending,
            "anomalies": [{"severity":"HIGH","amount":890,"category":"Travel","z_score":4.2,"date":"2025-02-01"}],
            "forecasts": [{"category":"Food & Dining","month":"2025-04","yhat":700,"yhat_lower":550,"yhat_upper":850}],
        }
        block = _build_context_block(context)
        assert "<financial_context>" in block
        assert "spending_summary" in block
        assert "anomalies" in block

        # Rule-based fallback works without API key
        coach = LLMCoach(api_key="", model="gemini-2.5-flash")
        result = coach.chat("What should I focus on?", context=context)
        assert "response" in result
        assert len(result["response"]) > 50
