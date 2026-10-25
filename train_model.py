
import pandas as pd
import json
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# File locations
data_path = Path("data/processed")
reports_path = Path("reports")
reports_path.mkdir(parents=True, exist_ok=True)

# Load the prepared data
X_train = pd.read_csv(data_path / "X_train.csv")
X_test = pd.read_csv(data_path / "X_test.csv")

y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

# Identify numeric and categorical columns
numeric_columns = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()

# Preprocess numeric columns
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocess categorical columns
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Build the complete machine learning pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

# Evaluate the model
metrics = {
    "accuracy": round(accuracy_score(y_test, predictions), 4),
    "precision": round(
        precision_score(y_test, predictions, zero_division=0), 4
    ),
    "recall": round(
        recall_score(y_test, predictions, zero_division=0), 4
    ),
    "f1_score": round(
        f1_score(y_test, predictions, zero_division=0), 4
    ),
    "roc_auc": round(roc_auc_score(y_test, probabilities), 4)
}

print("\n===== LOGISTIC REGRESSION RESULTS =====")
for name, value in metrics.items():
    print(f"{name}: {value}")

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(y_test, predictions, zero_division=0))

print("===== CONFUSION MATRIX =====")
print(confusion_matrix(y_test, predictions))

# Save evaluation metrics
with open(reports_path / "model_metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("\nMetrics saved to reports/model_metrics.json")
