import pandas as pd
import pytest

from mldk.io import prepare_X_for_predict, split_X_y


def test_split_X_y_missing_target():
    df = pd.DataFrame({"feature": [1, 2, 3]})
    with pytest.raises(ValueError, match="Target column 'target' not found"):
        split_X_y(df, "target")


def test_prepare_X_for_predict_drops_target():
    df = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    prepared = prepare_X_for_predict(df, "target")
    assert "target" not in prepared.columns
    assert prepared.shape[1] == 1
