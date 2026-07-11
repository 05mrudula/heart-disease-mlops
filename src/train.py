# ==========================================================
# Import libraries
# ==========================================================

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import GridSearchCV, train_test_split


# ==========================================================
# Project Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "heart_cleveland_cleaned.csv"

ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

MODEL_PATH = ARTIFACT_DIR / "heart_disease_random_forest.pkl"


# ==========================================================
# Load Dataset
# ==========================================================

df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape}")


# ==========================================================
# Prepare Features and Target
# ==========================================================

X = df.drop("target", axis=1)
y = df["target"]


# ==========================================================
# Train-Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# ==========================================================
# Hyperparameter Grid
# ==========================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}


# ==========================================================
# MLflow Experiment
# ==========================================================

mlflow.set_experiment("Heart Disease Prediction")

with mlflow.start_run():

    # Initialize model
    rf_model = RandomForestClassifier(random_state=42)

    # Grid Search
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )

    # Train model
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    print("Model training completed.")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    # ======================================================
    # Evaluate Model
    # ======================================================

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print("\nModel Performance")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # ======================================================
    # MLflow Logging
    # ======================================================

    mlflow.log_params(grid_search.best_params_)

    mlflow.log_metric("cv_accuracy", grid_search.best_score_)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Log trained model
    mlflow.sklearn.log_model(
    sk_model=best_model,
    name="model",
    input_example=X_test.iloc[:5]
    )

    # Save model locally
    joblib.dump(best_model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")

print("\nTraining pipeline completed successfully.")