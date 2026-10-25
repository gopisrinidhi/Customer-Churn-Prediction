
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Load the original training and test splits
data_path = Path("data/processed")
X_train = pd.read_csv(data_path / "X_train.csv")
X_test = pd.read_csv(data_path / "X_test.csv")
y_train = pd.read_csv(data_path / "y_train.csv").squeeze()
y_test = pd.read_csv(data_path / "y_test.csv").squeeze()

# Split training data into training and validation sets
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

numeric_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    include=["object", "str", "category"]
).columns.tolist()


def create_model():
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

    return Pipeline([
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


# Train on the training portion only
validation_model = create_model()
validation_model.fit(X_fit, y_fit)

# Select threshold using validation predictions
val_probabilities = validation_model.predict_proba(X_val)[:, 1]

thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
threshold_results = []

for threshold in thresholds:
    val_predictions = (val_probabilities >= threshold).astype(int)

    threshold_results.append({
        "threshold": threshold,
        "f1": f1_score(y_val, val_predictions, zero_division=0),
        "recall": recall_score(y_val, val_predictions, zero_division=0),
        "precision": precision_score(
            y_val, val_predictions, zero_division=0
        )
    })

best_result = max(threshold_results, key=lambda result: result["f1"])
best_threshold = best_result["threshold"]

print("===== VALIDATION THRESHOLD SELECTION =====")
print(f"Selected threshold: {best_threshold:.2f}")
print(f"Validation precision: {best_result['precision']:.4f}")
print(f"Validation recall: {best_result['recall']:.4f}")
print(f"Validation F1: {best_result['f1']:.4f}")

# Retrain the model on all training data
final_model = create_model()
final_model.fit(X_train, y_train)

# Evaluate on the held-out test set
test_probabilities = final_model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

print("\n===== FINAL TEST RESULTS =====")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy_score(y_test, test_predictions):.4f}")
print(
    f"Precision: "
    f"{precision_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(
    f"Recall: "
    f"{recall_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(
    f"F1 Score: "
    f"{f1_score(y_test, test_predictions, zero_division=0):.4f}"
)
print(f"ROC-AUC: {roc_auc_score(y_test, test_probabilities):.4f}")

print("\nClassification report:")
print(classification_report(y_test, test_predictions, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, test_predictions))

# Save the final evaluation summary
reports_path = Path("reports")
reports_path.mkdir(exist_ok=True)

summary = {
    "selected_threshold": best_threshold,
    "accuracy": accuracy_score(y_test, test_predictions),
    "precision": precision_score(
        y_test, test_predictions, zero_division=0
    ),
    "recall": recall_score(y_test, test_predictions, zero_division=0),
    "f1_score": f1_score(y_test, test_predictions, zero_division=0),
    "roc_auc": roc_auc_score(y_test, test_probabilities)
}

pd.DataFrame([summary]).to_csv(
    reports_path / "final_evaluation.csv", index=False
)

print("\nSaved to reports/final_evaluation.csv")
