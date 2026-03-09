"""Compute SHAP explanations for trained models.

Uses the appropriate SHAP explainer for each model type:
- LinearExplainer  → Logistic Regression
- TreeExplainer    → Random Forest, XGBoost
"""

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

from src.config import N_SHAP_BACKGROUND_SAMPLES, RANDOM_SEED


def _get_background_data(X_train: np.ndarray) -> np.ndarray:
    """Sample a background dataset for SHAP explainers.

    Parameters
    ----------
    X_train : np.ndarray
        Full training feature matrix.

    Returns
    -------
    np.ndarray
        Subsample used as the background distribution.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    n = min(N_SHAP_BACKGROUND_SAMPLES, X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n, replace=False)
    return X_train[idx]


def build_explainer(
    model,
    X_train: np.ndarray,
) -> shap.Explainer:
    """Build the appropriate SHAP explainer for a given model.

    Parameters
    ----------
    model : fitted estimator
        A trained sklearn or xgboost model.
    X_train : np.ndarray
        Training data (used as background for LinearExplainer).

    Returns
    -------
    shap.Explainer
        A SHAP explainer instance.
    """
    background = _get_background_data(X_train)

    if isinstance(model, LogisticRegression):
        return shap.LinearExplainer(model, background)
    else:
        # TreeExplainer for RandomForest, XGBoost
        return shap.TreeExplainer(model)


def compute_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
) -> np.ndarray:
    """Compute SHAP values for a set of samples.

    Parameters
    ----------
    explainer : shap.Explainer
        A fitted SHAP explainer.
    X : np.ndarray
        Feature matrix to explain.

    Returns
    -------
    np.ndarray
        SHAP values array of shape (n_samples, n_features).
        For binary classifiers that return 3-D arrays, the
        positive-class slice is returned.
    """
    shap_values = explainer.shap_values(X)

    # Some explainers return a list [class_0, class_1] or 3-D array
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # positive class
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    return shap_values


def compute_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> dict[str, np.ndarray]:
    """Derive global feature importance from SHAP values.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values of shape (n_samples, n_features).
    feature_names : list[str]
        Feature names matching the columns.

    Returns
    -------
    dict
        Keys:
        - mean_abs_shap: mean |SHAP| per feature (sorted descending)
        - feature_names_sorted: feature names in importance order
        - ranking: integer ranking (1 = most important)
    """
    mean_abs = np.abs(shap_values).mean(axis=0)

    sorted_idx = np.argsort(mean_abs)[::-1]
    feature_names_arr = np.array(feature_names)

    return {
        "mean_abs_shap": mean_abs[sorted_idx],
        "feature_names_sorted": feature_names_arr[sorted_idx].tolist(),
        "ranking": sorted_idx,
    }


def compute_baseline_shap(
    models: dict,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
    n_local_samples: int = 10,
) -> dict[str, dict]:
    """Compute baseline SHAP explanations for all models.

    Parameters
    ----------
    models : dict[str, model]
        Mapping of model name → fitted model.
    X_train : np.ndarray
        Training features (for background).
    X_test : np.ndarray
        Test features to explain.
    feature_names : list[str]
        Feature names.
    n_local_samples : int
        Number of test samples for local explanations.

    Returns
    -------
    dict[str, dict]
        Per-model dict with keys:
        - explainer: the SHAP explainer
        - shap_values_test: SHAP values for the full test set
        - shap_values_local: SHAP values for selected local samples
        - local_indices: indices of the selected samples
        - global_importance: output of compute_global_importance
    """
    rng = np.random.default_rng(RANDOM_SEED)
    local_idx = rng.choice(X_test.shape[0], size=n_local_samples, replace=False)

    results = {}
    for name, model in models.items():
        print(f"  Computing SHAP for {name}...")
        explainer = build_explainer(model, X_train)

        shap_values_test = compute_shap_values(explainer, X_test)
        shap_values_local = shap_values_test[local_idx]

        global_imp = compute_global_importance(shap_values_test, feature_names)

        results[name] = {
            "explainer": explainer,
            "shap_values_test": shap_values_test,
            "shap_values_local": shap_values_local,
            "local_indices": local_idx,
            "global_importance": global_imp,
        }

        top5 = global_imp["feature_names_sorted"][:5]
        print(f"    Top-5 features: {top5}")

    print("✓ Baseline SHAP explanations computed.")
    return results
