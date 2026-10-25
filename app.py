import plotly.express as px
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# Load saved model and input columns
MODEL_PATH = Path("models/churn_model.joblib")
COLUMNS_PATH = Path("models/model_columns.joblib")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    columns = joblib.load(COLUMNS_PATH)
    return model, columns

model, model_columns = load_model()

# Dashboard heading
st.title("📊 Customer Churn Prediction")
st.write(
    "Predict customer churn risk using a trained Machine Learning model."
)

st.info(
    "This tool estimates churn risk. Predictions support investigation "
    "and retention planning; they are not guarantees."
)

# Input form
st.header("Enter Customer Details")

with st.form("customer_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior_citizen = st.selectbox(
            "Senior Citizen", [0, 1],
            format_func=lambda value: "Yes" if value == 1 else "No"
        )
        partner = st.selectbox("Has Partner?", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service", ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security", ["Yes", "No", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup", ["Yes", "No", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection", ["Yes", "No", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support", ["Yes", "No", "No internet service"]
        )

    with col3:
        streaming_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"]
        )
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)", min_value=0.0,
            max_value=2000.0, value=70.0, step=1.0
        )
        total_charges = st.number_input(
            "Total Charges ($)", min_value=0.0,
            max_value=100000.0, value=840.0, step=10.0
        )

    submitted = st.form_submit_button(
        "Predict Churn Risk", type="primary"
    )

if submitted:
    # Build one input row with the exact source dataset columns
    customer = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    input_df = pd.DataFrame([customer])

    # Match the model's training column order
    input_df = input_df[model_columns]

    # Predict churn probability
    churn_probability = model.predict_proba(input_df)[0, 1]
    threshold = 0.55
    churn_prediction = int(churn_probability >= threshold)

    st.divider()
    st.header("Prediction Results")

    metric_col1, metric_col2 = st.columns(2)

    with metric_col1:
        st.metric(
            "Estimated Churn Probability",
            f"{churn_probability:.1%}"
        )

    with metric_col2:
        st.metric(
            "Decision Threshold",
            f"{threshold:.0%}"
        )

    if churn_prediction == 1:
        st.error("Higher predicted churn risk")
        st.write(
            "Consider reviewing this customer's needs and offering "
            "appropriate support or retention options."
        )
    else:
        st.success("Lower predicted churn risk")
        st.write(
            "Continue normal customer engagement and monitor future changes."
        )

    st.caption(
        "Risk classification uses a 0.55 threshold. The probability is "
        "a model estimate, not a guarantee that the customer will leave."
    )

# ==========================================
# BUSINESS ANALYTICS DASHBOARD
# ==========================================

st.divider()
st.title("📈 Customer Retention Analytics")

# Load original customer dataset
DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

@st.cache_data
def load_customer_data():
    data = pd.read_csv(DATA_PATH)
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"], errors="coerce"
    )
    return data

df = load_customer_data()

# Calculate business KPIs
total_customers = len(df)
churned_customers = (df["Churn"] == "Yes").sum()
retained_customers = (df["Churn"] == "No").sum()
churn_rate = churned_customers / total_customers * 100
average_monthly_charges = df["MonthlyCharges"].mean()

# Display KPI cards
st.subheader("Business Overview")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

kpi1.metric("Total Customers", f"{total_customers:,}")
kpi2.metric("Churned Customers", f"{churned_customers:,}")
kpi3.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
kpi4.metric(
    "Average Monthly Charges",
    f"${average_monthly_charges:.2f}"
)

st.caption(
    "These KPIs summarize the dataset. They are historical business "
    "metrics, not predictions for an individual customer."
)

# Churn distribution
st.subheader("Customer Churn Distribution")

churn_counts = (
    df["Churn"]
    .value_counts()
    .rename_axis("Customer Status")
    .reset_index(name="Number of Customers")
)

fig_churn = px.pie(
    churn_counts,
    names="Customer Status",
    values="Number of Customers",
    title="Customers Who Stayed vs. Churned",
    hole=0.45
)

st.plotly_chart(fig_churn, use_container_width=True)

# Churn rate by contract type
st.subheader("Churn Rate by Contract Type")

contract_summary = (
    df.groupby("Contract")
    .agg(
        Total_Customers=("Churn", "size"),
        Churned_Customers=(
            "Churn",
            lambda values: (values == "Yes").sum()
        )
    )
    .reset_index()
)

contract_summary["Churn Rate (%)"] = (
    contract_summary["Churned_Customers"]
    / contract_summary["Total_Customers"] * 100
)

fig_contract = px.bar(
    contract_summary,
    x="Contract",
    y="Churn Rate (%)",
    text=contract_summary["Churn Rate (%)"].round(1),
    title="Percentage of Customers Churning by Contract",
    hover_data=[
        "Total_Customers",
        "Churned_Customers"
    ]
)

fig_contract.update_traces(texttemplate="%{text}%", textposition="outside")
fig_contract.update_layout(yaxis_range=[0, 100])

st.plotly_chart(fig_contract, use_container_width=True)

# Monthly charges comparison
st.subheader("Monthly Charges: Churned vs. Retained")

fig_charges = px.box(
    df,
    x="Churn",
    y="MonthlyCharges",
    color="Churn",
    title="Monthly Charges Distribution by Customer Status",
    labels={
        "Churn": "Customer Churn",
        "MonthlyCharges": "Monthly Charges ($)"
    }
)

st.plotly_chart(fig_charges, use_container_width=True)

# Contract summary table
st.subheader("Contract Performance Summary")

contract_summary["Churn Rate (%)"] = (
    contract_summary["Churn Rate (%)"].round(2)
)

st.dataframe(
    contract_summary.sort_values(
        "Churn Rate (%)", ascending=False
    ),
    use_container_width=True,
    hide_index=True
)

