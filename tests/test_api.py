from fastapi.testclient import TestClient

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.app import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Heart Disease Prediction API is running."
    }


def test_predict():

    sample = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    response = client.post("/predict", json=sample)

    assert response.status_code == 200

    response_data = response.json()

    assert "prediction" in response_data
    assert "probability" in response_data

    assert response_data["prediction"] in [0, 1]
    assert 0.0 <= response_data["probability"] <= 1.0