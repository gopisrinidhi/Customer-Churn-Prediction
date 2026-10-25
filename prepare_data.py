
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# File paths
file_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

# Load raw dataset
data = pd.read_csv(file_path)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

# Remove customer ID: it identifies customers but isn't a useful
# predictive feature for this project.
data = data.drop(columns=["customerID"])

# Remove rows where the target is missing, if any
data = data.dropna(subset=["Churn"])

# Separate features and target
X = data.drop(columns=["Churn"])
y = data["Churn"].map({"No": 0, "Yes": 1})

# Split BEFORE learning preprocessing rules to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Save the split datasets for inspection and later use
X_train.to_csv(output_folder / "X_train.csv", index=False)
X_test.to_csv(output_folder / "X_test.csv", index=False)
y_train.to_csv(output_folder / "y_train.csv", index=False)
y_test.to_csv(output_folder / "y_test.csv", index=False)

print("Data preparation completed!")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training churn rate:", round(y_train.mean() * 100, 2), "%")
print("Testing churn rate:", round(y_test.mean() * 100, 2), "%")
print("Features:", X_train.shape[1])
print("Files saved in:", output_folder)
