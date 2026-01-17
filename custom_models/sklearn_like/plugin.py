"""Example plugin that uses the built-in sklearn workflow."""

from __future__ import annotations

from mldk.evaluate import run_evaluate
from mldk.predict import run_predict
from mldk.train import run_train


def train(
    train_csv: str,
    target: str,
    save_path: str,
    task: str,
    model: str,
    seed: int,
) -> str:
    """Train using the built-in pipeline and return the model path."""
    run_train(
        data_path=train_csv,
        target=target,
        save_path=save_path,
        task=task,
        model_name=model,
        seed=seed,
    )
    return save_path


def predict(
    data_csv: str,
    model_path: str,
    out_csv: str,
    id_col: str | None = None,
    target: str | None = None,
) -> None:
    """Generate predictions using the built-in pipeline."""
    run_predict(
        data_path=data_csv,
        model_path=model_path,
        out_path=out_csv,
        id_col=id_col,
    )


def evaluate(preds_csv: str, target: str, out_dir: str) -> None:
    """Evaluate predictions using the built-in metrics."""
    run_evaluate(preds_path=preds_csv, target_col=target, out_dir=out_dir)
