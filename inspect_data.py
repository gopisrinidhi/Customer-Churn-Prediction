
import pandas as pd
from pathlib import Path

# Locate the dataset
file_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Check whether the file exists
if not file_path.exists():
    print("Dataset not found!")
    print("Please check the CSV filename and data folder.")
else:
    # Load the dataset
    data = pd.read_csv(file_path)

    print("\n--- FIRST 5 ROWS ---")
    print(data.head())

    print("\n--- DATASET SHAPE ---")
    print(data.shape)

    print("\n--- COLUMN NAMES ---")
    print(data.columns.tolist())

    print("\n--- DATA TYPES ---")
    print(data.dtypes)

    print("\n--- MISSING VALUES ---")
    print(data.isnull().sum())

    print("\n--- DUPLICATE ROWS ---")
    print(data.duplicated().sum())

    print("\n--- CHURN DISTRIBUTION ---")
    print(data["Churn"].value_counts(dropna=False))
