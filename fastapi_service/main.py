"""
CareerCast - Milestone 3 FastAPI Service
---------------------------------------------------------------
A separate REST API (port 8000) alongside your Flask app (port 5000).
Reuses the SAME trained model files - no retraining needed.

Install:
    pip install "fastapi[standard]"

Run:
    fastapi dev fastapi_service/main.py
    (or: uvicorn fastapi_service.main:app --reload --port 8000)

Docs (automatic, free from FastAPI):
    http://127.0.0.1:8000/docs
"""

import pickle
import json
import re
import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="CareerCast API",
    description="Milestone 3 - REST endpoints for prediction, recommendation, and skill gap reports",
    version="1.0.0",
)

MODEL_DIR = "model"  # same folder your Flask app already uses


# ------------------------------------------------------------------
# Load the SAME trained artifacts your Flask app uses - no duplicate
# training, no separate model files needed.
# ------------------------------------------------------------------
def _load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found - train your models first (see career_pred_ml.ipynb)")
    with open(path, "rb") as f:
        return pickle.load(f)

tfidf = _load_pickle(f"{MODEL_DIR}/tfidf_vectorizer.pkl")
logreg_model = _load_pickle(f"{MODEL_DIR}/logreg_model.pkl")
rf_model = _load_pickle(f"{MODEL_DIR}/rf_model.pkl")
xgb_model = _load_pickle(f"{MODEL_DIR}/xgb_model.pkl")
label_encoder = _load_pickle(f"{MODEL_DIR}/label_encoder.pkl")

selector = None
if os.path.exists(f"{MODEL_DIR}/selector.pkl"):
    selector = _load_pickle(f"{MODEL_DIR}/selector.pkl")

with open(f"{MODEL_DIR}/metrics.json") as f:
    METRICS = json.load(f)


def clean(t):
    t = str(t).lower()
    t = re.sub(r'\d+', ' ', t)
    t = re.sub(r'[^a-z\s]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def vectorize(text_list):
    cleaned = [clean(t) for t in text_list]
    vec = tfidf.transform(cleaned)
    if selector is not None:
        vec = selector.transform(vec)
    return vec


def predict_with_best_model(vec):
    """Same multi-model confidence-based selection as your Flask app."""
    lr_probs = logreg_model.predict_proba(vec)[0]
    rf_probs = rf_model.predict_proba(vec)[0]
    xgb_probs_enc = xgb_model.predict_proba(vec)[0]

    lr_classes = logreg_model.classes_
    rf_classes = rf_model.classes_
    xgb_classes = label_encoder.inverse_transform(range(len(xgb_probs_enc)))

    candidates = [
        ("Logistic Regression", lr_probs, lr_classes),
        ("Random Forest", rf_probs, rf_classes),
        ("XGBoost", xgb_probs_enc, xgb_classes),
    ]
    best_name, best_probs, best_classes = max(candidates, key=lambda c: c[1].max())
    return best_name, best_probs, best_classes


# Same role-skill reference used by your Flask app - copy from app.py
# if you've expanded this list there.
ROLE_SKILL_MAP = {
    "Data Scientist": ["python", "machine learning", "statistics", "pandas", "numpy", "sql", "data visualization", "deep learning"],
    "Machine Learning Engineer": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "docker", "aws", "git"],
    "Data Analyst": ["sql", "excel", "power bi", "tableau", "data analysis", "statistics", "data visualization", "python"],
    "Frontend Developer": ["html", "css", "javascript", "react", "angular", "git", "rest api"],
    "Backend Developer": ["java", "python", "sql", "django", "flask", "rest api", "mongodb", "git"],
}


# ------------------------------------------------------------------
# Pydantic models = request/response schemas (FastAPI auto-validates
# and documents these for you - this is the big advantage over Flask)
# ------------------------------------------------------------------
class PredictRequest(BaseModel):
    resume_text: str

class PredictResponse(BaseModel):
    model_used: str
    predictions: List[dict]

class RecommendRequest(BaseModel):
    skills: List[str]
    top_k: Optional[int] = 5

class GapReportRequest(BaseModel):
    skills: List[str]
    target_role: str


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------
@app.get("/")
def root():
    return {"service": "CareerCast API", "status": "running", "docs": "/docs"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Milestone 3: prediction endpoint."""
    if not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text cannot be empty")

    vec = vectorize([req.resume_text])
    model_used, probs, classes = predict_with_best_model(vec)
    top3_idx = probs.argsort()[-3:][::-1]

    predictions = [
        {"category": str(classes[i]), "confidence": round(float(probs[i]) * 100, 1)}
        for i in top3_idx
    ]
    return {"model_used": model_used, "predictions": predictions}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    """Milestone 3: recommendation endpoint - Top-K roles by skill match."""
    skill_set = set(s.lower() for s in req.skills)
    results = []
    for role, required in ROLE_SKILL_MAP.items():
        required_set = set(required)
        matched = skill_set & required_set
        pct = round(len(matched) / len(required_set) * 100, 1) if required_set else 0
        results.append({"role": role, "match_percent": pct, "matched_skills": sorted(matched)})
    results.sort(key=lambda r: r["match_percent"], reverse=True)
    return {"recommendations": results[:req.top_k]}


@app.post("/gap-report")
def gap_report(req: GapReportRequest):
    """Milestone 3: skill gap report endpoint - reuses the same logic
    as your Flask /api/skill-gap route."""
    if req.target_role not in ROLE_SKILL_MAP:
        raise HTTPException(status_code=400, detail=f"Unknown role. Available: {list(ROLE_SKILL_MAP.keys())}")

    skill_set = set(s.lower() for s in req.skills)
    required_set = set(ROLE_SKILL_MAP[req.target_role])
    matched = skill_set & required_set
    missing = required_set - skill_set
    pct = round(len(matched) / len(required_set) * 100, 1) if required_set else 0

    suggestions = [f"Learn '{s}' - required for {req.target_role}" for s in sorted(missing)]

    return {
        "target_role": req.target_role,
        "match_percent": pct,
        "matched_skills": sorted(matched),
        "missing_skills": sorted(missing),
        "suggestions": suggestions,
    }


@app.get("/health")
def health():
    """Used by GitHub Actions CI later to check the service is alive."""
    return {"status": "ok", "accuracy": METRICS.get("accuracy")}
