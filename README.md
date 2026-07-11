# Heart Disease Prediction - MLOps Project

An end-to-end Machine Learning Operations (MLOps) project for predicting the presence of heart disease using a Random Forest Classifier. This project demonstrates the complete ML lifecycle—from model training and experiment tracking to API deployment, containerization, and Continuous Integration.

---

## Features

- Data preprocessing
- Random Forest Classifier
- Hyperparameter tuning using GridSearchCV
- MLflow experiment tracking
- Model serialization using Joblib
- FastAPI REST API
- Interactive Swagger UI
- Docker containerization
- GitHub Actions Continuous Integration (CI)

---

## Project Structure

```
heart-disease-mlops/
│
├── artifacts/
│   └── heart_disease_random_forest.pkl
│
├── data/
│   ├── heart_cleveland.csv
│   └── heart_cleveland_cleaned.csv
│
├── notebooks/
│   └── Heart_Disease_Prediction_ML_Model_Development.ipynb
│
├── src/
│   ├── train.py
│   └── app.py
│
├── tests/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the Cleveland Heart Disease Dataset from the UCI Machine Learning Repository.

Dataset includes patient attributes such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise Induced Angina
- ST Depression
- Slope
- Number of Major Vessels
- Thalassemia

Target:

- 0 → No Heart Disease
- 1 → Heart Disease

---

## Machine Learning Pipeline

1. Load dataset
2. Clean and preprocess data
3. Train/Test Split
4. Hyperparameter tuning using GridSearchCV
5. Train Random Forest model
6. Evaluate model
7. Log metrics and parameters using MLflow
8. Save trained model using Joblib

---

## Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | 0.9016 |
| Precision | 0.8667 |
| Recall | 0.9286 |
| F1 Score | 0.8966 |
| ROC-AUC | 0.9481 |

---

## MLflow

MLflow is used to track:

- Parameters
- Metrics
- Trained Models
- Experiments

Start MLflow UI:

```bash
mlflow ui
```

Open:

```
http://127.0.0.1:5000
```

---

## FastAPI

Start the API:

```bash
uvicorn src.app:app --reload
```

API available at:

```
http://127.0.0.1:8000
```

Swagger Documentation:

```
http://127.0.0.1:8000/docs
```

---

## API Endpoints

### GET /

Returns API status.

Example Response

```json
{
  "message": "Heart Disease Prediction API is running!"
}
```

---

### POST /predict

Predict heart disease.

Example Request

```json
{
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
```

Example Response

```json
{
  "prediction": 0
}
```

---

## Docker

Build Docker Image

```bash
docker build -t heart-disease-api .
```

Run Container

```bash
docker run -p 8000:8000 heart-disease-api
```

Open

```
http://localhost:8000/docs
```

---

## GitHub Actions

Continuous Integration workflow performs:

- Checkout Repository
- Setup Python
- Install Dependencies
- Execute Training Pipeline
- Verify FastAPI Application

Workflow file:

```
.github/workflows/ci.yml
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- FastAPI
- Uvicorn
- Joblib
- Docker
- GitHub Actions

---

## Future Improvements

- Kubernetes Deployment
- Cloud Deployment
- Monitoring
- Model Retraining Pipeline

---

## Author

**Mrudula Lingam**

GitHub: https://github.com/05mrudula

---

## License

This project is developed for learning and demonstration purposes.