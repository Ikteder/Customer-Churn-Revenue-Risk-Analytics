from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn and Revenue Risk Dashboard", layout="wide")
st.title("Customer Churn and Revenue Risk Dashboard")
st.caption("Review model performance, revenue-risk segments, and top at-risk customers.")

REPORTS = Path("reports")
FIGURES = REPORTS / "figures"

metrics_path = REPORTS / "metrics_summary.csv"
segments_path = REPORTS / "segment_risk_summary.csv"
risk_path = REPORTS / "top_revenue_risk_customers.csv"

if not metrics_path.exists():
    st.warning("Run the pipeline first: python -m src.pipeline --input data/raw/Telco-Customer-Churn.csv --output-dir reports")
    st.stop()

metrics_df = pd.read_csv(metrics_path)
st.subheader("Model comparison")
st.dataframe(metrics_df, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    if (FIGURES / "model_comparison.png").exists():
        st.image(str(FIGURES / "model_comparison.png"), caption="Model comparison")
with col2:
    if (FIGURES / "confusion_matrix_best_model.png").exists():
        st.image(str(FIGURES / "confusion_matrix_best_model.png"), caption="Confusion matrix")

st.subheader("Segment revenue risk")
if segments_path.exists():
    segment_df = pd.read_csv(segments_path)
    st.dataframe(segment_df.head(25), use_container_width=True)

st.subheader("Top revenue-risk customers")
if risk_path.exists():
    risk_df = pd.read_csv(risk_path)
    st.dataframe(risk_df.head(25), use_container_width=True)

st.subheader("Generated figures")
figure_names = [
    "churn_by_contract.png",
    "churn_by_tenure_bucket.png",
    "revenue_risk_by_segment.png",
    "roc_curves.png",
    "feature_importance_top15.png",
]
for fig_name in figure_names:
    fig_path = FIGURES / fig_name
    if fig_path.exists():
        st.image(str(fig_path), caption=fig_name)
