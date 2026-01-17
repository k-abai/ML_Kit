"""Evaluation entry points."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
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
from mldk.utils import now_ts


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


def _render_report(metrics: Dict[str, Any], row_count: int, target_col: str, task: str) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"- Dataset size: {row_count} rows",
        f"- Target column: {target_col}",
        f"- Inferred task: {task}",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Next Steps",
            "- Review feature quality and data coverage.",
            "- Compare against a simple baseline model.",
            "- Validate on a held-out dataset before deployment.",
        ]
    )
    return "\n".join(lines)


def run_evaluate(
    test_csv: str,
    model_path: str,
    target_col: str,
    out_dir: str,
    id_col: str | None = None,
) -> None:
    """Evaluate model predictions from a labeled test CSV."""
    df = pd.read_csv(test_csv)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in test data.")

    y_true = df[target_col]
    X = df.drop(columns=[target_col])

    if id_col and id_col in X.columns:
        X = X.drop(columns=[id_col])

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    y_pred = pipeline.predict(X)

    task = infer_task(y_true, "auto")
    metrics = _classification_metrics(y_true, y_pred) if task == "classification" else _regression_metrics(y_true, y_pred)

    run_dir = Path(out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    report_path = run_dir / "report.md"
    meta_path = run_dir / "meta.json"

    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = _render_report(metrics, len(df), target_col, task)
    report_path.write_text(report, encoding="utf-8")

    meta = {
        "timestamp": now_ts(),
        "n_rows": len(df),
        "inferred_task": task,
        "model_path": model_path,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[OK] Saved evaluation outputs to {run_dir}")
