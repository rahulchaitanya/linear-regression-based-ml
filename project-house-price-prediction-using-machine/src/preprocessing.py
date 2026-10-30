"""Dataset loading, validation, and feature preprocessing."""
from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

NUMERIC_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
CATEGORICAL_FEATURES = ["Location", "Condition", "Garage"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Price"
REQUIRED_COLUMNS = ["Id", *FEATURES, TARGET]


def load_dataset(path: Path) -> pd.DataFrame:
    """Read CSV and validate schema/value types, returning the untouched frame."""
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}. Place the CSV in data/.")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Could not read dataset at {path}: {exc}") from exc
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no records.")
    numeric = [*NUMERIC_FEATURES, TARGET]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    problems = []
    if frame[numeric].isna().any().any():
        problems.append("missing or non-numeric values in numeric columns")
    if frame[CATEGORICAL_FEATURES].isna().any().any():
        problems.append("missing categorical values")
    if (frame["Area"] <= 0).any(): problems.append("Area must be positive")
    for column in ["Bedrooms", "Bathrooms", "Floors"]:
        if (frame[column] < 0).any(): problems.append(f"{column} cannot be negative")
    if (frame["YearBuilt"] < 1800).any() or (frame["YearBuilt"] > 2026).any():
        problems.append("YearBuilt must be between 1800 and 2026")
    if (frame[TARGET] <= 0).any(): problems.append("Price must be positive")
    if problems:
        raise ValueError("Dataset validation failed: " + "; ".join(problems))
    return frame


def make_preprocessor() -> ColumnTransformer:
    """Create a transform fitted only inside the training pipeline."""
    return ColumnTransformer([
        ("numeric", "passthrough", NUMERIC_FEATURES),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
