from __future__ import annotations

import pandas as pd

YES_NO_SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ADD_ON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]


def _has_service(value: object) -> int:
    value_str = str(value).strip().lower()
    return int(value_str in {"yes", "fiber optic", "dsl"})


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()

    featured["ChurnLabel"] = featured["Churn"].map({"Yes": 1, "No": 0})
    featured["tenure_bucket"] = pd.cut(
        featured["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6 months", "7-12 months", "13-24 months", "25-48 months", "49-72 months"],
    )

    featured["service_count"] = featured[YES_NO_SERVICE_COLS].applymap(_has_service).sum(axis=1)
    featured["addon_count"] = featured[ADD_ON_COLS].applymap(_has_service).sum(axis=1)
    featured["is_month_to_month"] = (featured["Contract"] == "Month-to-month").astype(int)
    featured["is_auto_pay"] = featured["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    featured["estimated_annual_value"] = featured["MonthlyCharges"] * 12
    featured["six_month_revenue_at_risk"] = featured["MonthlyCharges"] * 6

    return featured
