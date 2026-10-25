
import joblib
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

# Load training data only
data_path = Path("data/processed")
X_train = pd.read_csv(data_path / "X_train.csv")
y_train = pd.read_csv(data_path / "y_train.csv").squeeze()

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ))
])

# Fit using all available training records
model.fit(X_train, y_train)

# Save model and expected input columns
models_path = Path("models")
models_path.mkdir(exist_ok=True)

joblib.dump(model, models_path / "churn_model.joblib")
joblib.dump(
    X_train.columns.tolist(),
    models_path / "model_columns.joblib"
)

print("Model saved successfully!")
print("File: models/churn_model.joblib")
print("Training records:", len(X_train))
print("Expected input features:", len(X_train.columns))
