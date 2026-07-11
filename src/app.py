# ==========================================================
# Import Libraries
# ==========================================================

import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ==========================================================
# Logging Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Load Trained Model
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_random_forest.pkl"

try:
    logger.info("Loading trained model...")

    model = joblib.load(MODEL_PATH)

    logger.info("Model loaded successfully.")
    logger.info("Heart Disease Prediction API started successfully.")
    logger.info(f"Model loaded from {MODEL_PATH}")

except Exception:
    logger.exception("Failed to load trained model.")
    raise


# ==========================================================
# Create FastAPI App
# ==========================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Predicts the presence of heart disease using a Random Forest model.",
    version="1.0"
)


# ==========================================================
# Input Schema
# ==========================================================

class HeartDiseaseInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ==========================================================
# Root Endpoint
# ==========================================================

@app.get("/")
def home():
    logger.info("Health check endpoint '/' accessed.")

    return {
        "message": "Heart Disease Prediction API is running."
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
def predict(data: HeartDiseaseInput):

    try:

        logger.info("Received prediction request.")

        input_df = pd.DataFrame([data.model_dump()])

        prediction = model.predict(input_df)[0]

        probability = model.predict_proba(input_df)[0]

        confidence = round(float(max(probability)), 4)

        logger.info(
            f"Prediction: {prediction}, Probability: {confidence}"
        )

        return {
            "prediction": int(prediction),
            "probability": confidence
        }

    except Exception:
        logger.exception("Prediction request failed.")
        raise