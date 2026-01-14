"""Input/output helpers for loading data."""

from __future__ import annotations

from typing import Tuple

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def split_X_y(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target series."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in training data.")
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


def prepare_X_for_predict(df: pd.DataFrame, target: str | None) -> pd.DataFrame:
    """Prepare a DataFrame for prediction by dropping the target column if present."""
    if target and target in df.columns:
        return df.drop(columns=[target])
    return df
