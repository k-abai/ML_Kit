"""Modeling helpers for task inference and pipeline creation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def infer_task(y: pd.Series, task: str) -> str:
    """Infer the task type from the target series when task is set to auto."""
    if task != "auto":
        return task

    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    if pd.api.types.is_numeric_dtype(y):
        y_non_null = y.dropna()
        if len(y_non_null) == 0:
            return "regression"
        values = y_non_null.to_numpy()
        integer_like = np.all(np.isclose(values, np.round(values)))
        if integer_like and y_non_null.nunique() <= 20:
            return "classification"

    return "regression"


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing ColumnTransformer for numeric and categorical columns."""
    numeric_features: List[str] = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features: List[str] = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []
    if numeric_features:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No features available for training.")

    return ColumnTransformer(transformers=transformers)


def build_model(task: str, model_name: str, seed: int) -> Tuple[object, str]:
    """Build a model for the specified task and model choice."""
    if model_name == "auto":
        model_name = "logreg" if task == "classification" else "ridge"

    if task == "classification":
        if model_name == "logreg":
            return LogisticRegression(max_iter=1000, random_state=seed), model_name
        if model_name == "rf":
            return RandomForestClassifier(random_state=seed), model_name
        raise ValueError("Unsupported classification model. Use logreg or rf.")

    if task == "regression":
        if model_name == "ridge":
            return Ridge(random_state=seed), model_name
        if model_name == "rf":
            return RandomForestRegressor(random_state=seed), model_name
        raise ValueError("Unsupported regression model. Use ridge or rf.")

    raise ValueError("Task must be classification or regression.")


def build_pipeline(preprocessor: ColumnTransformer, model: object) -> Pipeline:
    """Build a pipeline with preprocessing and model steps."""
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
