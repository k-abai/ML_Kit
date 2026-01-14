"""Training entry points."""

from __future__ import annotations

from typing import Dict

import joblib

from mldk.io import load_csv, split_X_y
from mldk.modeling import build_model, build_pipeline, build_preprocessor, infer_task
from mldk.utils import ensure_parent_dir


def run_train(
    train_path: str,
    target: str,
    out_path: str,
    task: str,
    model_name: str,
    seed: int,
) -> Dict[str, object]:
    """Train a model from CSV data and persist a joblib bundle."""
    df = load_csv(train_path)
    X, y = split_X_y(df, target)
    inferred_task = infer_task(y, task)
    preprocessor = build_preprocessor(X)
    model, resolved_model_name = build_model(inferred_task, model_name, seed)
    pipeline = build_pipeline(preprocessor, model)
    pipeline.fit(X, y)

    bundle = {
        "pipeline": pipeline,
        "task": inferred_task,
        "target": target,
        "model_name": resolved_model_name,
    }

    out_file = ensure_parent_dir(out_path)
    joblib.dump(bundle, out_file)
    print(f"[OK] Saved model to {out_file}")
    return bundle
