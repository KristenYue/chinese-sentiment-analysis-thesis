"""FastAPI服务入口。"""

from functools import lru_cache
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .classifier import SentimentClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_ARTIFACTS = (
    "xgboost_sentiment_model.joblib",
    "tfidf_vectorizer.joblib",
    "label_encoder.joblib",
)


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=200)


@lru_cache(maxsize=1)
def get_classifier() -> SentimentClassifier:
    configured = os.getenv("SENTIMENT_ARTIFACTS_DIR")
    if configured:
        artifacts_dir = Path(configured)
    else:
        retrained = PROJECT_ROOT / "artifacts" / "retrained"
        retrained_complete = all((retrained / name).exists() for name in REQUIRED_ARTIFACTS)
        artifacts_dir = retrained if retrained_complete else PROJECT_ROOT / "artifacts"
    return SentimentClassifier(artifacts_dir)


app = FastAPI(
    title="社交媒体评论情感分析API",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict[str, object]:
    classifier = get_classifier()
    return {
        "status": "ok",
        "labels": classifier.labels,
        "artifacts_dir": str(classifier.artifacts_dir),
        "feature_count": int(classifier.model.n_features_in_),
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    try:
        return get_classifier().predict(request.text).to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest) -> list[dict[str, object]]:
    try:
        return [result.to_dict() for result in get_classifier().predict_many(request.texts)]
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
