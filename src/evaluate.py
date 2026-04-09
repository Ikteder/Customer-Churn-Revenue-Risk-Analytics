from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_metrics_table(metrics_df: pd.DataFrame, output_dir: str | Path) -> None:
    out = _ensure_dir(output_dir)
    fig_dir = _ensure_dir(out / "figures")
    metrics_df.to_csv(out / "metrics_summary.csv", index=False)

    ranked = metrics_df.sort_values("roc_auc", ascending=False)
    plt.figure(figsize=(10, 5))
    plt.bar(ranked["model"], ranked["roc_auc"])
    plt.title("Model comparison by ROC-AUC")
    plt.ylabel("ROC-AUC")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(fig_dir / "model_comparison.png", dpi=200)
    plt.close()


def save_roc_curves(probability_frames: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = _ensure_dir(output_dir)
    fig_dir = _ensure_dir(out / "figures")

    plt.figure(figsize=(8, 6))
    for model_name, scores in probability_frames.items():
        if scores["y_prob"].isna().all():
            continue
        RocCurveDisplay.from_predictions(scores["y_true"], scores["y_prob"], name=model_name)
    plt.title("ROC curves by model")
    plt.tight_layout()
    plt.savefig(fig_dir / "roc_curves.png", dpi=200)
    plt.close()


def save_confusion_matrix(best_model_name: str, probability_frames: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    out = _ensure_dir(output_dir)
    fig_dir = _ensure_dir(out / "figures")
    scores = probability_frames[best_model_name]
    cm = confusion_matrix(scores["y_true"], scores["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f"Confusion Matrix - {best_model_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / "confusion_matrix_best_model.png", dpi=200)
    plt.close()


def save_feature_importance(best_model_name: str, fitted_pipelines: dict[str, object], output_dir: str | Path) -> None:
    out = _ensure_dir(output_dir)
    fig_dir = _ensure_dir(out / "figures")
    pipeline = fitted_pipelines[best_model_name]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        return

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
    )
    importance_df.to_csv(out / "feature_importance_top15.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["feature"], importance_df["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Top 15 Features - {best_model_name}")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance_top15.png", dpi=200)
    plt.close()
