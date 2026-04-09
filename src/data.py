from __future__ import annotations

import pandas as pd


def load_telco_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_telco_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    if "TotalCharges" in cleaned.columns:
        cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
        cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(
            cleaned["MonthlyCharges"] * cleaned["tenure"]
        )

    if "SeniorCitizen" in cleaned.columns:
        cleaned["SeniorCitizen"] = cleaned["SeniorCitizen"].astype(str)

    for col in cleaned.columns:
        if cleaned[col].dtype == object:
            cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned
