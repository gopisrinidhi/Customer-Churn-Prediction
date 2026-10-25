
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

# Load prepared data
data_path = Path("data/processed")
X_train = pd.read_csv(data_path / "X_train.csv")
X_test = pd.read_csv(data_path / "X_test.csv")
y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

# Identify columns
numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Preprocessing
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

# Train Random Forest
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

model.fit(X_train, y_train)

# Get churn probabilities
probabilities = model.predict_proba(X_test)[:, 1]

# Test several thresholds
results = []

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    predictions = (probabilities >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Accuracy": round(accuracy_score(y_test, predictions), 4),
        "Precision": round(
            precision_score(y_test, predictions, zero_division=0), 4
        ),
        "Recall": round(
            recall_score(y_test, predictions, zero_division=0), 4
        ),
        "F1 Score": round(
            f1_score(y_test, predictions, zero_division=0), 4
        )
    })

results_df = pd.DataFrame(results)

print("\n===== RANDOM FOREST THRESHOLD COMPARISON =====")
print(results_df.to_string(index=False))

reports_path = Path("reports")
reports_path.mkdir(exist_ok=True)
results_df.to_csv(
    reports_path / "threshold_comparison.csv", index=False
)

print("\nSaved to reports/threshold_comparison.csv")
