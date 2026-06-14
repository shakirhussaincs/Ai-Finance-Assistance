"""
pipelines/categorization_pipeline.py
Pipeline 1 — Transaction Categorization
  Primary  : TF-IDF + Logistic Regression  (runs on real data now)
  Ready for: DistilBERT fine-tune swap when GPU available

Trained on 4684 real labelled records from the two uploaded datasets.
"""
from __future__ import annotations
import os, sys, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

logger = logging.getLogger(__name__)


class CategorizationPipeline:
    """
    TF-IDF character n-gram + Logistic Regression classifier.
    Trained on real transaction data from personal & banking datasets.
    """

    def __init__(self):
        self.pipeline: SKPipeline | None = None
        self.label_encoder = LabelEncoder()
        self._fitted = False

    def _build(self) -> SKPipeline:
        return SKPipeline([
            ("tfidf", TfidfVectorizer(
                analyzer="char_wb",        # char n-grams handle typos & abbreviations
                ngram_range=(2, 4),
                max_features=25_000,
                sublinear_tf=True,
                strip_accents="unicode",
                lowercase=True,
            )),
            ("lr", LogisticRegression(
                C=5.0,
                max_iter=1000,
                solver="lbfgs",
                n_jobs=-1,
            )),
        ])

    #Training 

    def train(self, df: pd.DataFrame, eval: bool = True) -> Dict[str, Any]:
        """
        Train on DataFrame with columns: description (str), category (str).

        Returns full classification report dict.
        """
        texts  = df["description"].fillna("").astype(str).str.strip().str.lower().tolist()
        labels = self.label_encoder.fit_transform(df["category"].astype(str).tolist())

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )

        self.pipeline = self._build()
        self.pipeline.fit(X_train, y_train)
        self._fitted = True

        results = {"train_size": len(X_train), "test_size": len(X_test)}

        if eval:
            y_pred = self.pipeline.predict(X_test)
            report = classification_report(
                y_test, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0,
            )
            results["classification_report"] = report
            results["accuracy"] = report["accuracy"]

            # Cross-validation on full set
            cv_scores = cross_val_score(self._build(), texts, labels, cv=5, scoring="accuracy")
            results["cv_mean"] = float(cv_scores.mean())
            results["cv_std"]  = float(cv_scores.std())

            logger.info(f"[Categorizer] Accuracy: {report['accuracy']:.4f} | CV: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return results

    #Inference 

    def predict(self, descriptions: List[str]) -> List[Dict[str, Any]]:
        """
        Classify a list of merchant/description strings.

        Returns: [{category, confidence, top3: [{cat, prob}]}]
        """
        if not self._fitted:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        texts = [str(d).strip().lower() for d in descriptions]
        proba = self.pipeline.predict_proba(texts)
        idx   = np.argmax(proba, axis=1)

        results = []
        for j, i in enumerate(idx):
            # top-3 categories with probabilities
            top3_idx  = np.argsort(proba[j])[::-1][:3]
            top3      = [
                {"category": self.label_encoder.classes_[k], "probability": round(float(proba[j, k]), 4)}
                for k in top3_idx
            ]
            results.append({
                "description": descriptions[j],
                "category":    self.label_encoder.classes_[int(i)],
                "confidence":  round(float(proba[j, i]), 4),
                "top3":        top3,
                "model":       "tfidf_lr",
            })
        return results

    def predict_single(self, description: str) -> Dict[str, Any]:
        return self.predict([description])[0]

    #Persistence

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "le": self.label_encoder}, path)
        logger.info(f"[Categorizer] Saved → {path}")

    def load(self, path: str | Path):
        data = joblib.load(path)
        self.pipeline      = data["pipeline"]
        self.label_encoder = data["le"]
        self._fitted       = True
        logger.info(f"[Categorizer] Loaded ← {path}")

    @property
    def categories(self) -> List[str]:
        return list(self.label_encoder.classes_) if self._fitted else []


#Quick self-test 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    sys.path.insert(0, "..")
    from data_prep import build_combined

    data = build_combined(
        "dataset/11 march 2025.csv",
        "dataset/banking-transaction-categorization-dataset.csv",
    )
    train_df = data["categorization_train"]
    print(f"Training on {len(train_df)} labelled records …")

    pipe = CategorizationPipeline()
    results = pipe.train(train_df)

    print(f"\nAccuracy : {results['accuracy']:.4f}")
    print(f"CV Score : {results['cv_mean']:.4f} ± {results['cv_std']:.4f}")

    # Test on real examples from the dataset
    test_cases = [
        "Restuarant",        # personal dataset label
        "Coffe",             # personal dataset label
        "Uber trip CA3245",  # banking dataset description
        "Whole Foods Market San Francisco CA",
        "Netflix subscription",
        "ACME Payroll Direct Deposit",
        "PG&E electricity bill",
        "Taxi",
        "Market",
        "Sport",
    ]
    print("\n--- Predictions on test cases ---")
    preds = pipe.predict(test_cases)
    for p in preds:
        print(f"  '{p['description']:<40}' → {p['category']:<20} ({p['confidence']:.2f})")

    pipe.save("/home/claude/finance_coach/models/tfidf_categorizer.joblib")
    print("\nModel saved.")
