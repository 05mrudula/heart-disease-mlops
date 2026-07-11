from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# ==========================================================
# Load Trained Model
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_random_forest.pkl"

model = joblib.load(MODEL_PATH)


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
    return {
        "message": "Heart Disease Prediction API is running."
    }


# ==========================================================
# Prediction Endpoint
# ==========================================================

@app.post("/predict")
def predict(data: HeartDiseaseInput):

    input_df = pd.DataFrame([data.model_dump()])

    prediction = model.predict(input_df)[0]

    return {
        "prediction": int(prediction)
    }