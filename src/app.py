from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(title="Compliance Risk Scoring API", version="1.0.0")

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"

class Transaction(BaseModel):
    txn_amount: float = Field(..., ge=0, description="Transaction amount in USD")
    txn_count_24h: int = Field(..., ge=0, le=500, description="Number of txns in last 24h")
    account_age_days: int = Field(..., ge=0, le=5000, description="Account age in days")
    is_international: int = Field(..., ge=0, le=1, description="1 if international txn")
    device_change_7d: int = Field(..., ge=0, le=1, description="1 if device changed in last 7 days")
    failed_logins_24h: int = Field(..., ge=0, le=50, description="Failed logins in last 24h")

class Prediction(BaseModel):
    risk_score: float
    risk_label: str

class BatchRequest(BaseModel):
    items: List[Transaction]

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train it first: python -m src.train")
    return joblib.load(MODEL_PATH)

def featurize(t: Transaction) -> np.ndarray:
    return np.array([[
        t.txn_amount,
        t.txn_count_24h,
        t.account_age_days,
        t.is_international,
        t.device_change_7d,
        t.failed_logins_24h
    ]], dtype=float)

def label_from_score(score: float) -> str:
    if score >= 0.75:
        return "HIGH"
    if score >= 0.40:
        return "MEDIUM"
    return "LOW"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=Prediction)
def predict(txn: Transaction):
    model = load_model()
    x = featurize(txn)
    score = float(model.predict_proba(x)[0, 1])
    return {"risk_score": round(score, 4), "risk_label": label_from_score(score)}

@app.post("/predict_batch", response_model=List[Prediction])
def predict_batch(req: BatchRequest):
    model = load_model()
    X = np.vstack([featurize(t) for t in req.items])
    scores = model.predict_proba(X)[:, 1]
    return [{"risk_score": round(float(s), 4), "risk_label": label_from_score(float(s))} for s in scores]
