"""Prediction entry points."""

from __future__ import annotations

import joblib
import pandas as pd

from mldk.io import load_csv, prepare_X_for_predict
from mldk.utils import ensure_parent_dir


def run_predict(
    data_path: str,
    model_path: str,
    out_path: str,
    id_col: str | None = None,
) -> None:
    """Run predictions using a saved model bundle and save to CSV."""
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    target = bundle.get("target")

    df = load_csv(data_path)
    if id_col and id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in prediction data.")

    X = prepare_X_for_predict(df, target)
    preds = pipeline.predict(X)

    output = pd.DataFrame({"pred": preds})
    if id_col:
        output.insert(0, id_col, df[id_col])

    out_file = ensure_parent_dir(out_path)
    output.to_csv(out_file, index=False)
    print(f"[OK] Saved predictions to {out_file}")
