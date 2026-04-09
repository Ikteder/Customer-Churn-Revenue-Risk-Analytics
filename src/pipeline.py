from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from .analysis import save_business_summaries
from .data import clean_telco_data, load_telco_data
from .evaluate import save_confusion_matrix, save_feature_importance, save_metrics_table, save_roc_curves
from .features import add_business_features
from .modeling import train_and_compare_models


def save_business_figures(df: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    contract_summary = df.groupby("Contract", dropna=False)["ChurnLabel"].mean().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(contract_summary.index.astype(str), contract_summary.values)
    plt.title("Churn rate by contract type")
    plt.ylabel("Churn rate")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "churn_by_contract.png", dpi=200)
    plt.close()

    tenure_summary = df.groupby("tenure_bucket", dropna=False)["ChurnLabel"].mean()
    plt.figure(figsize=(9, 5))
    plt.bar(tenure_summary.index.astype(str), tenure_summary.values)
    plt.title("Churn rate by tenure bucket")
    plt.ylabel("Churn rate")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "churn_by_tenure_bucket.png", dpi=200)
    plt.close()

    segment_summary = df.groupby("Contract", dropna=False)["six_month_revenue_at_risk"].sum().sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(segment_summary.index.astype(str), segment_summary.values)
    plt.title("Six-month revenue at risk by contract type")
    plt.ylabel("Revenue at risk")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "revenue_risk_by_segment.png", dpi=200)
    plt.close()


def save_error_analysis(source_df: pd.DataFrame, X_test: pd.DataFrame, probability_frame: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    joined = source_df.loc[X_test.index].copy().join(probability_frame)

    false_positive_cases = joined[(joined["y_true"] == 0) & (joined["y_pred"] == 1)]
    false_negative_cases = joined[(joined["y_true"] == 1) & (joined["y_pred"] == 0)]

    false_positive_cases.to_csv(out / "false_positive_cases.csv", index=False)
    false_negative_cases.to_csv(out / "false_negative_cases.csv", index=False)


def save_top_revenue_risk_customers(source_df: pd.DataFrame, X_test: pd.DataFrame, probability_frame: pd.DataFrame, output_dir: str | Path, top_n: int = 50) -> None:
    out = Path(output_dir)
    joined = source_df.loc[X_test.index].copy().join(probability_frame)
    joined["expected_revenue_risk"] = joined["six_month_revenue_at_risk"] * joined["y_prob"]

    cols = ["customerID", "Contract", "PaymentMethod", "tenure", "MonthlyCharges", "six_month_revenue_at_risk", "y_prob", "expected_revenue_risk"]
    top_risk = joined.sort_values("expected_revenue_risk", ascending=False)[cols].head(top_n)
    top_risk.to_csv(out / "top_revenue_risk_customers.csv", index=False)


def run_pipeline(input_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_df = load_telco_data(input_path)
    cleaned_df = clean_telco_data(raw_df)
    featured_df = add_business_features(cleaned_df)
    featured_df.to_csv("data/processed/cleaned_telco_churn.csv", index=False)

    save_business_summaries(featured_df, out)
    save_business_figures(featured_df, out)

    training_outputs = train_and_compare_models(featured_df)
    save_metrics_table(training_outputs["metrics_df"], out)
    save_roc_curves(training_outputs["probability_frames"], out)
    save_confusion_matrix(training_outputs["best_model_name"], training_outputs["probability_frames"], out)
    save_feature_importance(training_outputs["best_model_name"], training_outputs["fitted_pipelines"], out)

    best_scores = training_outputs["probability_frames"][training_outputs["best_model_name"]]
    save_error_analysis(featured_df, training_outputs["X_test"], best_scores, out)
    save_top_revenue_risk_customers(featured_df, training_outputs["X_test"], best_scores, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn and revenue-risk analytics pipeline")
    parser.add_argument("--input", required=True, help="Path to raw Telco churn CSV")
    parser.add_argument("--output-dir", default="reports", help="Directory for saved outputs")
    args = parser.parse_args()
    run_pipeline(args.input, args.output_dir)
