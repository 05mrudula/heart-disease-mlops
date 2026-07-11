import joblib
import pandas as pd
from pathlib import Path


# Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "heart_disease_random_forest.pkl"
DATA_PATH = BASE_DIR / "data" / "heart_cleveland_cleaned.csv"


def test_model_file_exists():
    """Verify that the trained model file exists."""
    assert MODEL_PATH.exists(), "Model file does not exist."


def test_model_loads_successfully():
    """Verify that the model loads without errors."""
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_generates_prediction():
    """Verify that the loaded model can generate a valid prediction."""
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    X = df.drop("target", axis=1)

    prediction = model.predict(X.iloc[[0]])

    assert prediction[0] in [0, 1]