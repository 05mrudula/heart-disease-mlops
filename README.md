# Heart Disease Prediction - End-to-End MLOps Project

An end-to-end Machine Learning Operations (MLOps) project for predicting heart disease using Machine Learning. The project demonstrates the complete ML lifecycle, from data preprocessing and model training to experiment tracking, API development, automated testing, containerization, and Continuous Integration.

---

## Features

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Logistic Regression and Random Forest models
- Hyperparameter tuning using GridSearchCV
- Cross-validation and model evaluation
- MLflow experiment tracking
- Model serialization using Joblib
- FastAPI REST API
- Prediction confidence (probability)
- Interactive Swagger UI
- Postman API testing
- Automated unit testing with Pytest
- Code quality checks using Ruff
- Application logging and exception handling
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
│   ├── app.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── artifacts/
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

This project uses the **Cleveland Heart Disease Dataset** from the UCI Machine Learning Repository.

The dataset contains clinical attributes such as:

- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Resting ECG
- Maximum Heart Rate
- Exercise-Induced Angina
- ST Depression (Oldpeak)
- Slope
- Number of Major Vessels
- Thalassemia

Target:

- **0** → No Heart Disease
- **1** → Heart Disease

---

## Machine Learning Pipeline

1. Load dataset
2. Clean and preprocess data
3. Perform Exploratory Data Analysis
4. Train-Test Split
5. Train Logistic Regression and Random Forest models
6. Hyperparameter tuning using GridSearchCV
7. Evaluate model performance
8. Track experiments using MLflow
9. Save the best model using Joblib

---

## Model Performance

| Metric | Score |
|---------|-------:|
| Accuracy | 0.9016 |
| Precision | 0.8667 |
| Recall | 0.9286 |
| F1 Score | 0.8966 |
| ROC-AUC | 0.9481 |

---

## MLflow

MLflow is used to track:

- Hyperparameters
- Evaluation metrics
- Training runs
- Best performing model

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

API:

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
  "message": "Heart Disease Prediction API is running."
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
  "prediction": 0,
  "probability": 0.7452
}
```

---

## Testing

The project includes automated tests using **Pytest**.

Test coverage includes:

- API endpoint tests
- Model loading tests
- Model prediction tests
- Dataset validation tests

Run tests:

```bash
pytest
```

---

## Code Quality

Static code analysis is performed using **Ruff**.

Run Ruff:

```bash
ruff check .
```

---

## Docker

Build the Docker image:

```bash
docker build -t heart-disease-api .
```

Run the container:

```bash
docker run -p 8000:8000 heart-disease-api
```

Open:

```
http://localhost:8000/docs
```

---

## GitHub Actions

Continuous Integration pipeline performs:

- Checkout repository
- Setup Python
- Install dependencies
- Run Ruff linting
- Execute model training pipeline
- Run Pytest test suite
- Verify FastAPI application

Workflow:

```
.github/workflows/ci.yml
```

---

## Logging

Application logging is implemented for:

- Model loading
- API startup
- Prediction requests
- Prediction results
- Training pipeline
- Error handling

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
- Pytest
- Ruff
- Docker
- GitHub Actions
- Postman

---

## Future Improvements

- Cloud deployment
- Kubernetes deployment
- Model monitoring
- Automated model retraining
- Model registry
- Performance monitoring dashboard

---

## Author

**Mrudula Lingam**

GitHub: https://github.com/05mrudula

---

## License

This project is developed for learning and educational purposes.