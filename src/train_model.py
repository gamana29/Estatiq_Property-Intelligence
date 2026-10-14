
import os
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Project paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "train.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "models")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

if "SalePrice" not in df.columns:
    raise ValueError(
        "Target column 'SalePrice' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

df = df.dropna(subset=["SalePrice"]).copy()
X = df.drop(columns=["SalePrice"])
y = df["SalePrice"]

# Remove identifier columns when present
id_columns = [
    col for col in X.columns
    if col.lower() in ["id", "pid", "parcelid"]
]
X = X.drop(columns=id_columns, errors="ignore")

# Identify feature types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Preprocess numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# Split data before training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
}

results = []
best_model = None
best_rmse = float("inf")
best_name = None

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE:  {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"R²:   {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = pipeline
        best_name = name

# Save comparison and best model
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv(
    os.path.join(RESULTS_DIR, "model_comparison.csv"),
    index=False
)

joblib.dump(
    best_model,
    os.path.join(MODEL_DIR, "best_model.joblib")
)

print("\nModel comparison:")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_name}")
print(f"Saved model to: {os.path.join(MODEL_DIR, 'best_model.joblib')}")
print("Saved comparison to: results/model_comparison.csv")
