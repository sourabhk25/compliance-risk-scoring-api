# Compliance Risk Scoring API (FastAPI + Scikit-learn)

A small, production-oriented prototype that trains a risk-scoring model on **synthetic transactional data** and serves real-time predictions via **FastAPI**.

## What this project does
- Generates synthetic transaction records with a configurable risk signal
- Trains a **Logistic Regression** model using **Scikit-learn**
- Serves predictions through REST endpoints:
  - `GET /health` – service health check  
  - `POST /predict` – risk scoring for a single transaction  
  - `POST /predict_batch` – batch risk scoring  

## Tech Stack
Python, FastAPI, Pydantic, Pandas, NumPy, Scikit-learn, Uvicorn, Joblib

---

## Quickstart (Local)

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
