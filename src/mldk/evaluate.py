"""Evaluation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

from mldk.modeling import infer_task
from mldk.utils import ensure_parent_dir, now_ts


def _classification_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    labels = y_true.dropna().unique()
    average = "binary" if len(labels) <= 2 else "macro"
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def _regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _render_report(metrics: Dict[str, Any], row_count: int, task: str) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Dataset size: {row_count} rows",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Next steps",
            "- Review feature quality and consider additional signal.",
            "- Compare with a stronger baseline model.",
            "- Validate on a held-out dataset before deployment.",
        ]
    )
    return "\n".join(lines)


def run_evaluate(preds_path: str, target_col: str, out_dir: str) -> None:
    """Evaluate predictions and write metrics, report, and metadata."""
    df = pd.read_csv(preds_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in predictions file.")
    if "pred" not in df.columns:
        raise ValueError("Prediction column 'pred' not found in predictions file.")

    y_true = df[target_col]
    y_pred = df["pred"]
    task = infer_task(y_true, "auto")

    metrics = _classification_metrics(y_true, y_pred) if task == "classification" else _regression_metrics(y_true, y_pred)

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = ensure_parent_dir(run_dir / "metrics.json")
    report_path = ensure_parent_dir(run_dir / "report.md")
    meta_path = ensure_parent_dir(run_dir / "meta.json")

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = _render_report(metrics, len(df), task)
    report_path.write_text(report, encoding="utf-8")

    meta = {"timestamp": now_ts(), "task": task, "row_count": len(df)}
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Saved evaluation outputs to {run_dir}")
