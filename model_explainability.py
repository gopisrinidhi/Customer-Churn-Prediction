
import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Model Explainability",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Model Explainability")
st.write(
    "Understand which customer features the Random Forest "
    "model relies on when predicting churn."
)

FILE_PATH = Path("reports/feature_importance.csv")

if not FILE_PATH.exists():
    st.error(
        "Feature importance file not found. "
        "Please run feature_importance.py first."
    )
    st.stop()

importance_df = pd.read_csv(FILE_PATH)

# Normalize column names for easier matching
importance_df.columns = [
    col.strip().lower().replace(" ", "_")
    for col in importance_df.columns
]

feature_col = next(
    (col for col in importance_df.columns
     if col in ["feature", "feature_name", "features"]),
    None
)

importance_col = next(
    (col for col in importance_df.columns
     if col in ["importance", "feature_importance", "importance_score"]),
    None
)

if feature_col is None or importance_col is None:
    st.error("Could not identify the feature and importance columns.")
    st.write("Columns found:", list(importance_df.columns))
    st.stop()

importance_df = importance_df.rename(
    columns={
        feature_col: "Feature",
        importance_col: "Importance"
    }
)

importance_df["Importance"] = pd.to_numeric(
    importance_df["Importance"], errors="coerce"
)
importance_df = importance_df.dropna(
    subset=["Feature", "Importance"]
)

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

if importance_df.empty:
    st.warning("No valid feature importance values were found.")
    st.stop()

top_n = min(15, len(importance_df))
top_features = importance_df.head(top_n).sort_values(
    "Importance", ascending=True
)

st.subheader("Top Features Used by the Model")

fig = px.bar(
    top_features,
    x="Importance",
    y="Feature",
    orientation="h",
    title=f"Top {top_n} Feature Importances",
    labels={
        "Importance": "Importance Score",
        "Feature": "Customer Feature"
    },
    hover_data={"Importance": ":.4f"}
)

st.plotly_chart(fig, use_container_width=True)

st.subheader("Feature Importance Table")
st.dataframe(
    importance_df.head(top_n).reset_index(drop=True),
    use_container_width=True,
    hide_index=True
)

st.subheader("How to Interpret This")

st.markdown("""
- **Higher importance:** The model relied more on that feature
  across its trees.
- **Lower importance:** The feature contributed less to the
  model's overall importance ranking.
- **Important limitation:** Importance does not show whether a
  feature increases or decreases an individual customer's risk.
- **Association is not causation:** These rankings do not prove
  that a feature causes customers to leave.
""")

st.caption(
    "These scores describe the trained Random Forest model. "
    "They are not causal explanations or guarantees for an individual prediction."
)
