"""Train predictive models for the Adult Income classification task.

Models:
- Logistic Regression (interpretable baseline)
- Random Forest (black-box)
- XGBoost (black-box)

All hyperparameters are read from src.config.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.config import (
    LOGISTIC_REGRESSION_PARAMS,
    RANDOM_FOREST_PARAMS,
    XGBOOST_PARAMS,
)


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **override_params,
) -> LogisticRegression:
    """Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features.
    y_train : np.ndarray
        Training labels.
    **override_params
        Any parameters to override from the default config.

    Returns
    -------
    LogisticRegression
        Fitted model.
    """
    params = {**LOGISTIC_REGRESSION_PARAMS, **override_params}
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **override_params,
) -> RandomForestClassifier:
    """Train a Random Forest classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features.
    y_train : np.ndarray
        Training labels.
    **override_params
        Any parameters to override from the default config.

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    params = {**RANDOM_FOREST_PARAMS, **override_params}
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    **override_params,
) -> XGBClassifier:
    """Train an XGBoost classifier.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features.
    y_train : np.ndarray
        Training labels.
    **override_params
        Any parameters to override from the default config.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    params = {**XGBOOST_PARAMS, **override_params}
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> dict[str, LogisticRegression | RandomForestClassifier | XGBClassifier]:
    """Train all three models and return them in a dict.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features.
    y_train : np.ndarray
        Training labels.

    Returns
    -------
    dict[str, model]
        Mapping of model name to fitted model.
    """
    models = {}

    print("  Training Logistic Regression...")
    models["LogisticRegression"] = train_logistic_regression(X_train, y_train)

    print("  Training Random Forest...")
    models["RandomForest"] = train_random_forest(X_train, y_train)

    print("  Training XGBoost...")
    models["XGBoost"] = train_xgboost(X_train, y_train)

    print(f"✓ All {len(models)} models trained.")
    return models
