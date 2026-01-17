"""CLI for training, prediction, evaluation, and run workflows."""

from __future__ import annotations

import argparse
from pathlib import Path
from types import ModuleType

import joblib
import pandas as pd

from mldk.evaluate import run_evaluate
from mldk.io import load_csv, prepare_X_for_predict
from mldk.predict import run_predict
from mldk.train import run_train
from mldk.utils import ensure_parent_dir, load_plugin


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="ML diagnostics kit CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a model.")
    train_parser.add_argument("--data", required=True, help="Path to training CSV.")
    train_parser.add_argument("--target", required=True, help="Target column name.")
    train_parser.add_argument("--save", required=True, help="Path to save the model joblib.")
    train_parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Task type (default: auto).",
    )
    train_parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "logreg", "rf", "ridge"],
        help="Model choice (default: auto).",
    )
    train_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_parser.add_argument("--custom", help="Path to custom plugin directory.")

    predict_parser = subparsers.add_parser("predict", help="Generate predictions.")
    predict_parser.add_argument("--data", required=True, help="Path to prediction CSV.")
    predict_parser.add_argument("--model", required=True, help="Path to saved model joblib.")
    predict_parser.add_argument("--out", required=True, help="Output CSV for predictions.")
    predict_parser.add_argument("--id-col", help="Optional ID column for output.")
    predict_parser.add_argument("--custom", help="Path to custom plugin directory.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate predictions.")
    evaluate_parser.add_argument("--preds", required=True, help="Path to predictions CSV.")
    evaluate_parser.add_argument("--target", required=True, help="Target column in predictions.")
    evaluate_parser.add_argument("--out", required=True, help="Output run directory.")
    evaluate_parser.add_argument("--custom", help="Path to custom plugin directory.")

    run_parser = subparsers.add_parser("run", help="Train, predict, and evaluate.")
    run_parser.add_argument("--train", required=True, help="Training CSV path.")
    run_parser.add_argument("--test", required=True, help="Test CSV path.")
    run_parser.add_argument("--target", required=True, help="Target column name.")
    run_parser.add_argument("--out", required=True, help="Run directory.")
    run_parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Task type (default: auto).",
    )
    run_parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "logreg", "rf", "ridge"],
        help="Model choice (default: auto).",
    )
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    run_parser.add_argument("--custom", help="Path to custom plugin directory.")

    return parser


def _run_full_pipeline(args: argparse.Namespace, plugin: ModuleType | None) -> None:
    run_dir = Path(args.out)
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    preds_path = run_dir / "preds.csv"

    if plugin:
        returned_model = plugin.train(
            train_csv=args.train,
            target=args.target,
            save_path=str(model_path),
            task=args.task,
            model=args.model,
            seed=args.seed,
        )
        resolved_model_path = str(returned_model) if returned_model else str(model_path)
        plugin.predict(
            data_csv=args.test,
            model_path=resolved_model_path,
            out_csv=str(preds_path),
            id_col=None,
            target=args.target,
        )
        plugin.evaluate(preds_csv=str(preds_path), target=args.target, out_dir=str(run_dir))
        return

    run_train(
        data_path=args.train,
        target=args.target,
        save_path=str(model_path),
        task=args.task,
        model_name=args.model,
        seed=args.seed,
    )

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    target = bundle.get("target")

    test_df = load_csv(args.test)
    X = prepare_X_for_predict(test_df, target)
    preds = pipeline.predict(X)

    output = pd.DataFrame({"pred": preds})
    if target and target in test_df.columns:
        output.insert(0, target, test_df[target])

    out_file = ensure_parent_dir(preds_path)
    output.to_csv(out_file, index=False)
    print(f"[OK] Saved predictions to {out_file}")

    if target and target in output.columns:
        run_evaluate(preds_path=str(preds_path), target_col=target, out_dir=str(run_dir))


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    plugin = load_plugin(args.custom) if getattr(args, "custom", None) else None

    if args.command == "train":
        if plugin:
            plugin.train(
                train_csv=args.data,
                target=args.target,
                save_path=args.save,
                task=args.task,
                model=args.model,
                seed=args.seed,
            )
            return
        run_train(
            data_path=args.data,
            target=args.target,
            save_path=args.save,
            task=args.task,
            model_name=args.model,
            seed=args.seed,
        )
        return

    if args.command == "predict":
        if plugin:
            plugin.predict(
                data_csv=args.data,
                model_path=args.model,
                out_csv=args.out,
                id_col=args.id_col,
                target=None,
            )
            return
        run_predict(
            data_path=args.data,
            model_path=args.model,
            out_path=args.out,
            id_col=args.id_col,
        )
        return

    if args.command == "evaluate":
        if plugin:
            plugin.evaluate(preds_csv=args.preds, target=args.target, out_dir=args.out)
            return
        run_evaluate(preds_path=args.preds, target_col=args.target, out_dir=args.out)
        return

    if args.command == "run":
        _run_full_pipeline(args, plugin)
        return


if __name__ == "__main__":
    main()
