import pandas as pd
from src.features import add_business_features


def test_add_business_features_creates_expected_columns():
    df = pd.DataFrame(
        {
            "customerID": ["0001"],
            "Churn": ["Yes"],
            "tenure": [5],
            "Contract": ["Month-to-month"],
            "PaymentMethod": ["Electronic check"],
            "MonthlyCharges": [70.0],
            "PhoneService": ["Yes"],
            "MultipleLines": ["No"],
            "OnlineSecurity": ["No"],
            "OnlineBackup": ["Yes"],
            "DeviceProtection": ["No"],
            "TechSupport": ["No"],
            "StreamingTV": ["Yes"],
            "StreamingMovies": ["No"],
        }
    )

    featured = add_business_features(df)
    expected_cols = {
        "ChurnLabel",
        "tenure_bucket",
        "service_count",
        "addon_count",
        "is_month_to_month",
        "is_auto_pay",
        "estimated_annual_value",
        "six_month_revenue_at_risk",
    }
    assert expected_cols.issubset(featured.columns)
