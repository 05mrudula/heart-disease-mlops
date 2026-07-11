import pandas as pd
from pathlib import Path


# Define dataset path
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "heart_cleveland_cleaned.csv"


def test_dataset_exists():
    """Verify that the cleaned dataset exists."""
    assert DATA_PATH.exists(), "Dataset file does not exist."


def test_dataset_not_empty():
    """Verify that the dataset is not empty."""
    df = pd.read_csv(DATA_PATH)

    assert len(df) > 0


def test_no_missing_values():
    """Verify that the cleaned dataset contains no missing values."""
    df = pd.read_csv(DATA_PATH)

    assert df.isnull().sum().sum() == 0


def test_target_column_exists():
    """Verify that the target column is present."""
    df = pd.read_csv(DATA_PATH)

    assert "target" in df.columns


def test_expected_number_of_features():
    """Verify that the dataset contains the expected number of columns."""
    df = pd.read_csv(DATA_PATH)

    # 13 features + 1 target column
    assert len(df.columns) == 14