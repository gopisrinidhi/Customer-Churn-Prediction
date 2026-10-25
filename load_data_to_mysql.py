
import pandas as pd
import pymysql
from pathlib import Path
from getpass import getpass

# Dataset location
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Read CSV
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"], errors="coerce"
)

# MySQL credentials
username = input("MySQL username (usually root): ").strip()
password = getpass("MySQL password: ")
host = input("MySQL host (press Enter for localhost): ").strip()

if not host:
    host = "localhost"

connection = None
cursor = None

try:
    connection = pymysql.connect(
        host=host,
        user=username,
        password=password,
        database="customer_churn_db",
        charset="utf8mb4",
        autocommit=False
    )

    cursor = connection.cursor()

    # Create table with columns matching the CSV
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customerID VARCHAR(30) PRIMARY KEY,
            gender VARCHAR(20),
            SeniorCitizen TINYINT,
            Partner VARCHAR(10),
            Dependents VARCHAR(10),
            tenure INT,
            PhoneService VARCHAR(10),
            MultipleLines VARCHAR(30),
            InternetService VARCHAR(30),
            OnlineSecurity VARCHAR(30),
            OnlineBackup VARCHAR(30),
            DeviceProtection VARCHAR(30),
            TechSupport VARCHAR(30),
            StreamingTV VARCHAR(30),
            StreamingMovies VARCHAR(30),
            Contract VARCHAR(30),
            PaperlessBilling VARCHAR(10),
            PaymentMethod VARCHAR(50),
            MonthlyCharges DECIMAL(10,2),
            TotalCharges DECIMAL(10,2),
            Churn VARCHAR(10)
        )
    """)

    # Replace existing rows so rerunning doesn't duplicate data
    cursor.execute("DELETE FROM customers")

    columns = list(df.columns)
    column_sql = ", ".join(f"`{col}`" for col in columns)
    placeholders = ", ".join(["%s"] * len(columns))

    insert_sql = (
        f"INSERT INTO customers ({column_sql}) "
        f"VALUES ({placeholders})"
    )

    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append(tuple(
            None if pd.isna(value) else value
            for value in row
        ))

    cursor.executemany(insert_sql, rows)
    connection.commit()

    cursor.execute("SELECT COUNT(*) FROM customers")
    count = cursor.fetchone()[0]

    print("\nImport completed successfully!")
    print("CSV rows:", len(df))
    print("MySQL rows:", count)

except Exception as error:
    if connection:
        connection.rollback()
    print("\nImport failed:", error)

finally:
    if cursor:
        cursor.close()
    if connection:
        connection.close()
