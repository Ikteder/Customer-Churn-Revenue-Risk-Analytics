from __future__ import annotations

from pathlib import Path
import pandas as pd


def save_business_summaries(df: pd.DataFrame, output_dir: str | Path) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    churn_by_contract = (
        df.groupby("Contract", dropna=False)
        .agg(
            customers=("customerID", "count"),
            churn_rate=("ChurnLabel", "mean"),
            avg_monthly_charge=("MonthlyCharges", "mean"),
            revenue_risk=("six_month_revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values("churn_rate", ascending=False)
    )

    churn_by_tenure = (
        df.groupby("tenure_bucket", dropna=False)
        .agg(
            customers=("customerID", "count"),
            churn_rate=("ChurnLabel", "mean"),
            revenue_risk=("six_month_revenue_at_risk", "sum"),
        )
        .reset_index()
    )

    segment_risk = (
        df.groupby(["Contract", "PaymentMethod"], dropna=False)
        .agg(
            customers=("customerID", "count"),
            churn_rate=("ChurnLabel", "mean"),
            avg_monthly_charge=("MonthlyCharges", "mean"),
            six_month_revenue_at_risk=("six_month_revenue_at_risk", "sum"),
        )
        .reset_index()
        .sort_values(["six_month_revenue_at_risk", "churn_rate"], ascending=[False, False])
    )

    churn_by_contract.to_csv(out / "churn_by_contract_summary.csv", index=False)
    churn_by_tenure.to_csv(out / "churn_by_tenure_summary.csv", index=False)
    segment_risk.to_csv(out / "segment_risk_summary.csv", index=False)
