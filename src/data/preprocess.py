"""Preprocessing pipeline for the Adult Income dataset.

Builds a scikit-learn ColumnTransformer that independently handles
numeric (scaling) and categorical (one-hot encoding) features.
Provides helpers for train/test splitting and saving processed data.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    DATA_PROCESSED_DIR,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET_COLUMN,
    TEST_SIZE,
)


def build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer for the Adult Income dataset.

    Returns
    -------
    ColumnTransformer
        Transformer with StandardScaler for numeric features and
        OneHotEncoder for categorical features.
    """
    numeric_transformer = Pipeline(
        steps=[("scaler", StandardScaler())],
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False),
            ),
        ],
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drop any columns not listed
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a DataFrame into train/test, separating features and target.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset including the target column.
    test_size : float
        Fraction of data reserved for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    # Encode target as binary int
    y = (df[TARGET_COLUMN] == ">50K").astype(int)
    X = df.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def fit_transform_data(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Fit the preprocessor on training data and transform both splits.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessing pipeline.
    X_train : pd.DataFrame
        Training features (raw).
    X_test : pd.DataFrame
        Test features (raw).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        X_train_processed, X_test_processed, feature_names
    """
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out().tolist()
    return X_train_processed, X_test_processed, feature_names


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: list[str],
) -> None:
    """Save processed arrays and metadata to data/processed/.

    Parameters
    ----------
    X_train : np.ndarray
        Processed training features.
    X_test : np.ndarray
        Processed test features.
    y_train : pd.Series
        Training labels.
    y_test : pd.Series
        Test labels.
    feature_names : list[str]
        Names of the processed features.
    """
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    np.save(DATA_PROCESSED_DIR / "X_train.npy", X_train)
    np.save(DATA_PROCESSED_DIR / "X_test.npy", X_test)
    np.save(DATA_PROCESSED_DIR / "y_train.npy", y_train.values)
    np.save(DATA_PROCESSED_DIR / "y_test.npy", y_test.values)

    pd.Series(feature_names).to_csv(
        DATA_PROCESSED_DIR / "feature_names.csv",
        index=False,
        header=False,
    )
    print(f"✓ Processed data saved to {DATA_PROCESSED_DIR}")


def run_preprocessing_pipeline(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, list[str], ColumnTransformer]:
    """Execute the full preprocessing pipeline end-to-end.

    Parameters
    ----------
    df : pd.DataFrame
        Raw combined dataset from load_data.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test, feature_names, fitted_preprocessor
    """
    X_train_raw, X_test_raw, y_train, y_test = split_data(df)
    preprocessor = build_preprocessor()
    X_train, X_test, feature_names = fit_transform_data(
        preprocessor,
        X_train_raw,
        X_test_raw,
    )
    save_processed_data(X_train, X_test, y_train, y_test, feature_names)

    print(f"  Training set:  {X_train.shape[0]:,} rows × {X_train.shape[1]} features")
    print(f"  Test set:      {X_test.shape[0]:,} rows × {X_test.shape[1]} features")

    return X_train, X_test, y_train, y_test, feature_names, preprocessor
