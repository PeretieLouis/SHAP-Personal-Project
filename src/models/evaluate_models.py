"""Evaluate trained models and export results.

Computes accuracy, ROC-AUC, and classification reports.
Saves structured results to outputs/metrics/.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

from src.config import METRICS_DIR


def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """Evaluate a single model on the test set.

    Parameters
    ----------
    model : fitted sklearn/xgboost estimator
        The trained model.
    X_test : np.ndarray
        Processed test features.
    y_test : np.ndarray
        Test labels.
    model_name : str
        Human-readable model name.

    Returns
    -------
    dict
        Keys: model, accuracy, roc_auc.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"  {model_name:<25s}  Accuracy: {accuracy:.4f}  ROC-AUC: {roc_auc:.4f}")

    return {
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
    }


def evaluate_all_models(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """Evaluate all models and return a summary DataFrame.

    Parameters
    ----------
    models : dict[str, model]
        Mapping of model name to fitted model.
    X_test : np.ndarray
        Processed test features.
    y_test : np.ndarray
        Test labels.

    Returns
    -------
    pd.DataFrame
        One row per model with accuracy and ROC-AUC.
    """
    results = []
    for name, model in models.items():
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)

    df_results = pd.DataFrame(results)
    return df_results


def save_evaluation_results(df_results: pd.DataFrame) -> None:
    """Save the evaluation DataFrame to CSV.

    Parameters
    ----------
    df_results : pd.DataFrame
        Model evaluation results.
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / "model_evaluation.csv"
    df_results.to_csv(path, index=False)
    print(f"✓ Evaluation results saved to {path}")


def print_classification_reports(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print detailed classification reports for all models.

    Parameters
    ----------
    models : dict[str, model]
        Mapping of model name to fitted model.
    X_test : np.ndarray
        Processed test features.
    y_test : np.ndarray
        Test labels.
    """
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{'=' * 60}")
        print(f"Classification Report — {name}")
        print(f"{'=' * 60}")
        print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
