"""CLI for training and prediction."""

from __future__ import annotations

import argparse

from mldk.predict import run_predict
from mldk.train import run_train


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="ML diagnostics kit CLI")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--train", help="Path to training CSV.")
    mode_group.add_argument("--predict", help="Path to prediction CSV.")
    mode_group.add_argument("--evaluate", help="Path to labeled test CSV for evaluation.")


    parser.add_argument("--target", help="Target column name for training.")
    parser.add_argument("--input", help="Input column name for evaluation.")
    parser.add_argument("--out", required=True, help="Output path for model or predictions.")
    parser.add_argument("--model-path", help="Path to saved model joblib.")
    parser.add_argument(
        "--task",
        default="auto",
        choices=["auto", "classification", "regression"],
        help="Task type (default: auto).",
    )
    parser.add_argument(
        "--model",
        default="auto",
        choices=["auto", "logreg", "rf", "ridge"],
        help="Model choice (default: auto).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--id-col", help="Optional ID column for prediction output.")
    return parser


def main() -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args()

    if args.train:
        if not args.target:
            parser.error("--target is required when using --train.")
        run_train(
            train_path=args.train,
            target=args.target,
            out_path=args.out,
            task=args.task,
            model_name=args.model,
            seed=args.seed,
        )
        return

    if args.predict:
        if not args.model_path:
            parser.error("--model-path is required when using --predict.")
        run_predict(
            predict_path=args.predict,
            model_path=args.model_path,
            out_path=args.out,
            id_col=args.id_col,
        )
        return
    
    if args.evaluate:
        from mldk.evaluate import run_evaluate
        if not args.model_path:
            parser.error("--model-path is required when using --predict.")
        run_evaluate(
            test_path=args.evaluate,
            model_path=args.model_path,
            target_col=args.target,
            input_col=args.input,
            out_dir=args.out,
            id_col=args.id_col,
        )



if __name__ == "__main__":
    main()
