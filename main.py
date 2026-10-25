from pathlib import Path

import joblib
import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
FEATURES_PATH = BASE_DIR / "models" / "feature_columns.json"

app = FastAPI(
    title="BCG Customer Churn Prediction API",
    description="API for predicting customer churn using the trained ML model.",
    version="1.0.0",
)


class PredictionRequest(BaseModel):
    records: list[dict[str, Any]]


def load_artifacts():
    if not MODEL_PATH.exists() or not FEATURES_PATH.exists():
        raise RuntimeError(
            "Model files are missing. Run train_model.py first to create "
            "models/churn_model.joblib and models/feature_columns.json."
        )

    model = joblib.load(MODEL_PATH)
    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)
    return model, features


@app.get("/")
def home():
    return {
        "message": "BCG Customer Churn Prediction API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_available": MODEL_PATH.exists() and FEATURES_PATH.exists(),
    }


@app.get("/features")
def get_features():
    try:
        _, features = load_artifacts()
        return {"features": features, "count": len(features)}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict")
def predict(request: PredictionRequest):
    if not request.records:
        raise HTTPException(status_code=400, detail="Provide at least one record.")

    try:
        model, features = load_artifacts()
        input_df = pd.DataFrame(request.records)

        missing = [col for col in features if col not in input_df.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Required model features are missing.",
                    "missing_features": missing,
                },
            )

        input_df = input_df[features]
        predictions = model.predict(input_df)

        probabilities = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[:, 1].tolist()

        results = []
        for index, prediction in enumerate(predictions):
            item = {
                "prediction": int(prediction),
                "churn_label": "Yes" if int(prediction) == 1 else "No",
            }
            if probabilities is not None:
                item["churn_probability"] = round(float(probabilities[index]), 4)
            results.append(item)

        return {"count": len(results), "predictions": results}

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
