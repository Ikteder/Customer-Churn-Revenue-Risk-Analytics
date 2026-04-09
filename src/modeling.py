from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None

TARGET_COL = "ChurnLabel"
DROP_COLS = ["Churn", "customerID"]


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns] + [TARGET_COL])
    y = df[TARGET_COL]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor, numeric_cols, categorical_cols


def get_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced"),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric="logloss",
        )

    return models


def train_and_compare_models(df: pd.DataFrame, random_state: int = 42):
    X, y = split_features_target(df)
    preprocessor, _, _ = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    metrics_rows = []
    fitted_pipelines = {}
    probability_frames = {}

    for model_name, estimator in get_models(random_state=random_state).items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

        metrics_rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan,
                "f1": f1_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
            }
        )

        fitted_pipelines[model_name] = pipeline
        probability_frames[model_name] = pd.DataFrame(
            {"y_true": y_test.to_numpy(), "y_pred": y_pred, "y_prob": y_prob if y_prob is not None else np.nan},
            index=X_test.index,
        )

    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    best_model_name = metrics_df.iloc[0]["model"]

    return {
        "X_test": X_test,
        "y_test": y_test,
        "metrics_df": metrics_df,
        "fitted_pipelines": fitted_pipelines,
        "probability_frames": probability_frames,
        "best_model_name": best_model_name,
    }
