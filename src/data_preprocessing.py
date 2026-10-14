
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
OUTPUT_DIR = PROJECT_ROOT / "data"


def load_data():
    """Load the housing dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Place your housing CSV at data/train.csv."
        )

    df = pd.read_csv(DATA_PATH)

    if "SalePrice" not in df.columns:
        raise ValueError(
            "The dataset must contain a 'SalePrice' target column."
        )

    return df


def prepare_data():
    """Separate features and target, then create train/test sets."""
    df = load_data()

    # Remove rows without a known sale price.
    df = df.dropna(subset=["SalePrice"])

    # Remove an identifier column if present.
    df = df.drop(columns=["Id"], errors="ignore")

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Keep the test set separate from model training.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Number of features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()
