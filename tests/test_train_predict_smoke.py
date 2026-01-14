from pathlib import Path

import pandas as pd

from mldk.predict import run_predict
from mldk.train import run_train


def test_train_predict_smoke(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[1]
    train_csv = repo_root / "examples" / "data" / "train_example.csv"
    predict_csv = repo_root / "examples" / "data" / "predict_example.csv"

    model_path = tmp_path / "model.joblib"
    preds_path = tmp_path / "preds.csv"

    run_train(
        data_path=str(train_csv),
        target="target",
        save_path=str(model_path),
        task="auto",
        model_name="auto",
        seed=42,
    )

    run_predict(
        data_path=str(predict_csv),
        model_path=str(model_path),
        out_path=str(preds_path),
        id_col=None,
    )

    preds = pd.read_csv(preds_path)
    predict_df = pd.read_csv(predict_csv)

    assert preds_path.exists()
    assert len(preds) == len(predict_df)
    assert "pred" in preds.columns
