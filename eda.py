
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# File paths
file_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
output_folder = Path("reports/eda")
output_folder.mkdir(parents=True, exist_ok=True)

# Load data
data = pd.read_csv(file_path)

# Fix TotalCharges, which may contain blank strings
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"], errors="coerce"
)

sns.set_theme(style="whitegrid")

# 1. Dataset overview
print("Dataset shape:", data.shape)
print("\nChurn distribution:")
print(data["Churn"].value_counts())

print("\nChurn percentage:")
print((data["Churn"].value_counts(normalize=True) * 100).round(2))

# 2. Missing values
print("\nMissing values:")
print(data.isnull().sum()[data.isnull().sum() > 0])

# 3. Churn distribution chart
plt.figure(figsize=(7, 5))
sns.countplot(data=data, x="Churn")
plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig(output_folder / "churn_distribution.png", dpi=150)
plt.show()

# 4. Tenure comparison
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="Churn", y="tenure")
plt.title("Customer Tenure vs Churn")
plt.tight_layout()
plt.savefig(output_folder / "tenure_vs_churn.png", dpi=150)
plt.show()

# 5. Monthly charges comparison
plt.figure(figsize=(8, 5))
sns.boxplot(data=data, x="Churn", y="MonthlyCharges")
plt.title("Monthly Charges vs Churn")
plt.tight_layout()
plt.savefig(output_folder / "monthly_charges_vs_churn.png", dpi=150)
plt.show()

# 6. Contract analysis
plt.figure(figsize=(9, 5))
sns.countplot(data=data, x="Contract", hue="Churn")
plt.title("Churn by Contract Type")
plt.tight_layout()
plt.savefig(output_folder / "contract_vs_churn.png", dpi=150)
plt.show()

# 7. Churn rate by contract
contract_churn = pd.crosstab(
    data["Contract"],
    data["Churn"],
    normalize="index"
) * 100

print("\nChurn percentage by contract:")
print(contract_churn.round(2))

print("\nEDA completed. Charts saved in reports/eda/")
