"""
app.py
------
FastAPI backend for the Phishing Website Detection System.

Endpoints:
    GET  /          → serves static/index.html
    POST /predict   → main detection endpoint (ENHANCED)

IMPROVEMENTS in this version:
    • Logging of every request and prediction
    • Explainable AI (XAI) – top 2 most important features returned
    • Risk level label: Low / Medium / High
    • Rule engine now BOOSTS probability instead of forcing phishing
    • 3 new ML features (has_https, url_entropy, special_char_count)
    • Stronger URL validation
"""

import os
import pickle
import logging
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
#from fastapi.responses import HTMLResponse
#from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel
from typing import List

from utils.preprocess import clean_url, is_valid_url
from utils.features import extract_features
from utils.rules import run_rules

# ──────────────────────────────────────────────
# NEW FEATURE: Simple Python logging setup
# ──────────────────────────────────────────────

# VIVA EXPLANATION:
# Logging is essential in production systems to:
#   1. Debug unexpected predictions without stopping the server.
#   2. Audit which URLs were scanned (useful for security teams).
#   3. Track patterns of abuse over time (many requests from one IP, etc.)
# We use Python's built-in 'logging' module — no extra libraries needed.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),                       # print to console
        logging.FileHandler("predictions.log"),        # write to file
    ],
)
logger = logging.getLogger("phishing_detector")


# ──────────────────────────────────────────────
# Feature names (must match train_model.py + features.py)
# ──────────────────────────────────────────────

# NEW FEATURE: Used for XAI – maps feature index → human-readable name
FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_digits",
    "has_suspicious_keyword",
    "subdomain_count",
    "has_https",           # NEW
    "url_entropy",         # NEW
    "special_char_count",  # NEW
]


# ──────────────────────────────────────────────
# App initialisation
# ──────────────────────────────────────────────

app = FastAPI(
    title="Phishing Detector API",
    description="Detects phishing URLs via ML + rule-based hybrid system.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
#templates = Jinja2Templates(directory="templates")

# ──────────────────────────────────────────────
# Load ML model
# ──────────────────────────────────────────────

MODEL_PATH = Path(__file__).parent / "model" / "model.pkl"

def _load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(
            "Model file not found. Run `python train_model.py` first."
        )
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

model = _load_model()
logger.info("✅ Model loaded successfully from %s", MODEL_PATH)


# ──────────────────────────────────────────────
# Pydantic schemas  (IMPROVED)
# ──────────────────────────────────────────────

class URLRequest(BaseModel):
    url: str


class PredictionResponse(BaseModel):
    url: str
    prediction: str          # "Phishing" | "Legitimate" | "Uncertain" | "Invalid Input"
    probability: float       # probability of being phishing (0.0 – 1.0)
    risk_level: str          # NEW FEATURE: "Low Risk" | "Medium Risk" | "High Risk"
    rule_triggered: str      # name of triggered rule(s) or "None"
    top_features: List[str]  # NEW FEATURE (XAI): top 2 most important features


# ──────────────────────────────────────────────
# Confidence thresholds
# ──────────────────────────────────────────────

PHISHING_THRESHOLD   = 0.65
LEGITIMATE_THRESHOLD = 0.35


def _label_from_probability(prob: float) -> str:
    """Map phishing probability to a prediction label."""
    if prob >= PHISHING_THRESHOLD:
        return "Phishing"
    if prob <= LEGITIMATE_THRESHOLD:
        return "Legitimate"
    return "Uncertain"


# NEW FEATURE: Risk level based on probability
def _risk_level(prob: float) -> str:
    """
    Return a human-readable risk level string.

    # VIVA EXPLANATION:
    # Raw probabilities (e.g. 0.7231) are hard for non-technical users to interpret.
    # Risk levels give an instant, clear signal:
    #   Low Risk    → probability < 0.35   (likely safe)
    #   Medium Risk → 0.35 ≤ prob < 0.65   (uncertain, proceed carefully)
    #   High Risk   → probability ≥ 0.65   (likely phishing)
    """
    if prob >= PHISHING_THRESHOLD:
        return "High Risk"
    if prob <= LEGITIMATE_THRESHOLD:
        return "Low Risk"
    return "Medium Risk"


# NEW FEATURE: Explainable AI helper
def _get_top_features(n: int = 2) -> List[str]:
    """
    Return the names of the top-n most important features from the trained model.

    Uses model.feature_importances_ — a built-in RandomForest attribute that
    measures how much each feature reduces impurity across all decision trees.

    # VIVA EXPLANATION:
    # Feature importances tell us WHICH signals the model relies on most.
    # If "url_length" is #1 and "url_entropy" is #2, it means the model learned
    # that long, random-looking URLs are the strongest indicators of phishing.
    # This builds trust in the model and helps us improve it in future versions.
    """
    importances = model.feature_importances_
    top_indices = importances.argsort()[::-1][:n]   # descending order
    return [FEATURE_NAMES[i] for i in top_indices]


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def serve_index():
    return FileResponse("static/index.html")
##@app.get("/", include_in_schema=False)
#async def serve_index(request: Request):
    """Serve the frontend HTML page."""
    index_path = STATIC_DIR / "index.html"
    return FileResponse(str(index_path))
#@app.get("/", response_class=HTMLResponse)
#async def serve_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: URLRequest):
    """
    IMPROVED Hybrid phishing detection pipeline:
        1. Preprocess (clean) the URL.
        2. Validate the URL (enhanced checks).
        3. Run rule-based checks → accumulate probability BOOST (not force).
        4. Extract 9 ML features (including 3 new ones) and get model probability.
        5. Apply rule boost to ML probability (capped at 1.0).
        6. Determine label, risk level, and top XAI features.
        7. Log the result.
    """

    # ── Step 1: Preprocess ──────────────────────────────────────────────────
    raw_url = request.url
    clean   = clean_url(raw_url)

    # NEW FEATURE: Log incoming request
    logger.info("🔍 Incoming URL: %s", raw_url)

    # ── Step 2: Validate ────────────────────────────────────────────────────
    if not is_valid_url(clean):
        logger.warning("⚠️  Invalid URL rejected: %s", raw_url)
        return PredictionResponse(
            url=raw_url,
            prediction="Invalid Input",
            probability=0.0,
            risk_level="N/A",
            rule_triggered="None",
            top_features=[],
        )

    # ── Step 3: Rule-based check (IMPROVED LOGIC – boost, not force) ────────
    # run_rules now returns (boost_amount, list_of_triggered_rule_names)
    rule_boost, triggered_names = run_rules(clean)

    # ── Step 4: ML prediction ────────────────────────────────────────────────
    # Pass raw_url so has_https() can check the original scheme
    features      = extract_features(clean, raw_url=raw_url)
    feature_array = np.array(features).reshape(1, -1)

    proba      = model.predict_proba(feature_array)[0]
    phish_prob = float(proba[1])   # probability of being phishing

    # ── Step 5: Merge rule boost + ML probability ────────────────────────────
    # IMPROVED LOGIC: Rules BOOST the probability, not override it.
    # This prevents a single rule from incorrectly labelling legitimate domains.
    # VIVA EXPLANATION:
    #   Old: if rule fired → phish_prob = max(phish_prob, 0.80) [hard override]
    #   New: phish_prob = min(phish_prob + boost, 1.0)           [soft nudge]
    #   Example: ML=0.45, 1 rule fires → 0.45+0.20=0.65 → "Phishing"
    #   Example: ML=0.10, 1 rule fires → 0.10+0.20=0.30 → still "Legitimate"
    if rule_boost > 0:
        phish_prob = min(phish_prob + rule_boost, 1.0)   # IMPROVED LOGIC

    rule_summary = ", ".join(triggered_names) if triggered_names else "None"

    # ── Step 6: Final labels ──────────────────────────────────────────────────
    prediction  = _label_from_probability(phish_prob)
    risk        = _risk_level(phish_prob)

    # NEW FEATURE: XAI – top 2 most important features
    top_feats = _get_top_features(n=2)

    # ── Step 7: Log result ────────────────────────────────────────────────────
    # NEW FEATURE: structured log line for every prediction
    logger.info(
        "📊 Result | URL=%-50s | Pred=%-10s | Prob=%.4f | Risk=%-11s | Rules=%s",
        clean, prediction, phish_prob, risk, rule_summary
    )

    return PredictionResponse(
        url=clean,
        prediction=prediction,
        probability=round(phish_prob, 4),
        risk_level=risk,
        rule_triggered=rule_summary,
        top_features=top_feats,
    )
