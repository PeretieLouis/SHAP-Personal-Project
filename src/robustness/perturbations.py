"""Perturbation experiments for SHAP robustness evaluation.

Applies controlled perturbations to the training data or model,
retrains, and recomputes SHAP explanations. Each perturbation
produces a set of SHAP values that can be compared to the baseline.
"""

import numpy as np

from src.config import (
    BOOTSTRAP_SAMPLE_FRAC,
    GAUSSIAN_NOISE_STD,
    N_PERTURBATION_RUNS,
    PERTURBATION_SEEDS,
    TRAINING_FRACTIONS,
)
from src.explainability.compute_shap import (
    build_explainer,
    compute_global_importance,
    compute_shap_values,
)
from src.models.train_models import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)

# Map model names to their training functions
_TRAIN_FN = {
    "LogisticRegression": train_logistic_regression,
    "RandomForest": train_random_forest,
    "XGBoost": train_xgboost,
}


def _retrain_and_explain(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    local_indices: np.ndarray,
    feature_names: list[str],
    **train_overrides,
) -> dict:
    """Retrain a model and compute SHAP explanations.

    Parameters
    ----------
    model_name : str
        One of 'LogisticRegression', 'RandomForest', 'XGBoost'.
    X_train : np.ndarray
        (Possibly perturbed) training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features (always the same).
    local_indices : np.ndarray
        Indices of the local test samples.
    feature_names : list[str]
        Feature names.
    **train_overrides
        Keyword arguments to override default model params.

    Returns
    -------
    dict
        Keys: shap_values_test, shap_values_local, global_importance.
    """
    train_fn = _TRAIN_FN[model_name]
    model = train_fn(X_train, y_train, **train_overrides)

    explainer = build_explainer(model, X_train)
    shap_vals = compute_shap_values(explainer, X_test)
    shap_local = shap_vals[local_indices]
    global_imp = compute_global_importance(shap_vals, feature_names)

    return {
        "shap_values_test": shap_vals,
        "shap_values_local": shap_local,
        "global_importance": global_imp,
    }


# ──────────────────────────────────────────────
# Perturbation 1: Model random seed variation
# ──────────────────────────────────────────────
def run_seed_perturbation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    local_indices: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """Retrain with different random seeds, keeping data identical.

    Parameters
    ----------
    model_name : str
        Model identifier.
    X_train, y_train : np.ndarray
        Original training data (unchanged).
    X_test : np.ndarray
        Test features.
    local_indices : np.ndarray
        Indices for local explanations.
    feature_names : list[str]
        Feature names.

    Returns
    -------
    list[dict]
        One result dict per seed.
    """
    results = []
    for i, seed in enumerate(PERTURBATION_SEEDS):
        print(f"      seed run {i + 1}/{N_PERTURBATION_RUNS} (seed={seed})")
        res = _retrain_and_explain(
            model_name,
            X_train,
            y_train,
            X_test,
            local_indices,
            feature_names,
            random_state=seed,
        )
        results.append(res)
    return results


# ──────────────────────────────────────────────
# Perturbation 2: Bootstrap resampling
# ──────────────────────────────────────────────
def run_bootstrap_perturbation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    local_indices: np.ndarray,
    feature_names: list[str],
) -> list[dict]:
    """Retrain on bootstrap-resampled training data.

    Parameters
    ----------
    model_name : str
        Model identifier.
    X_train, y_train : np.ndarray
        Original training data.
    X_test : np.ndarray
        Test features.
    local_indices : np.ndarray
        Indices for local explanations.
    feature_names : list[str]
        Feature names.

    Returns
    -------
    list[dict]
        One result dict per bootstrap run.
    """
    n_samples = int(X_train.shape[0] * BOOTSTRAP_SAMPLE_FRAC)
    results = []
    for i, seed in enumerate(PERTURBATION_SEEDS):
        print(f"      bootstrap run {i + 1}/{N_PERTURBATION_RUNS}")
        rng = np.random.default_rng(seed)
        idx = rng.choice(X_train.shape[0], size=n_samples, replace=True)
        X_boot, y_boot = X_train[idx], y_train[idx]

        res = _retrain_and_explain(
            model_name,
            X_boot,
            y_boot,
            X_test,
            local_indices,
            feature_names,
        )
        results.append(res)
    return results


# ──────────────────────────────────────────────
# Perturbation 3: Gaussian noise on numeric features
# ──────────────────────────────────────────────
def run_noise_perturbation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    local_indices: np.ndarray,
    feature_names: list[str],
    n_numeric_features: int,
) -> list[dict]:
    """Retrain after adding Gaussian noise to numeric features.

    Noise std = GAUSSIAN_NOISE_STD × feature std (computed from training data).
    Only the first `n_numeric_features` columns are perturbed (the numeric
    columns come first in the ColumnTransformer output).

    Parameters
    ----------
    model_name : str
        Model identifier.
    X_train, y_train : np.ndarray
        Original training data.
    X_test : np.ndarray
        Test features.
    local_indices : np.ndarray
        Indices for local explanations.
    feature_names : list[str]
        Feature names.
    n_numeric_features : int
        Number of leading columns that are numeric.

    Returns
    -------
    list[dict]
        One result dict per noise run.
    """
    # Compute per-feature std for numeric columns
    feature_stds = X_train[:, :n_numeric_features].std(axis=0)

    results = []
    for i, seed in enumerate(PERTURBATION_SEEDS):
        print(f"      noise run {i + 1}/{N_PERTURBATION_RUNS}")
        rng = np.random.default_rng(seed)
        noise = rng.normal(
            loc=0.0,
            scale=GAUSSIAN_NOISE_STD * feature_stds,
            size=(X_train.shape[0], n_numeric_features),
        )
        X_noisy = X_train.copy()
        X_noisy[:, :n_numeric_features] += noise

        res = _retrain_and_explain(
            model_name,
            X_noisy,
            y_train,
            X_test,
            local_indices,
            feature_names,
        )
        results.append(res)
    return results


# ──────────────────────────────────────────────
# Perturbation 4: Training dataset size variation
# ──────────────────────────────────────────────
def run_size_perturbation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    local_indices: np.ndarray,
    feature_names: list[str],
) -> dict[float, dict]:
    """Retrain on reduced fractions of the training data.

    Parameters
    ----------
    model_name : str
        Model identifier.
    X_train, y_train : np.ndarray
        Original training data.
    X_test : np.ndarray
        Test features.
    local_indices : np.ndarray
        Indices for local explanations.
    feature_names : list[str]
        Feature names.

    Returns
    -------
    dict[float, dict]
        Mapping of fraction → result dict.
    """
    results = {}
    for frac in TRAINING_FRACTIONS:
        n = int(X_train.shape[0] * frac)
        print(f"      size={frac:.0%} ({n:,} rows)")
        rng = np.random.default_rng(PERTURBATION_SEEDS[0])
        idx = rng.choice(X_train.shape[0], size=n, replace=False)
        X_sub, y_sub = X_train[idx], y_train[idx]

        res = _retrain_and_explain(
            model_name,
            X_sub,
            y_sub,
            X_test,
            local_indices,
            feature_names,
        )
        results[frac] = res
    return results


# ──────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────
def run_all_perturbations(
    models: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    local_indices: np.ndarray,
    feature_names: list[str],
    n_numeric_features: int,
) -> dict[str, dict[str, list[dict] | dict[float, dict]]]:
    """Run all perturbation experiments for all models.

    Parameters
    ----------
    models : dict[str, model]
        Mapping of model name → fitted baseline model.
    X_train, y_train : np.ndarray
        Original training data.
    X_test : np.ndarray
        Test features.
    local_indices : np.ndarray
        Indices for local explanations (from baseline).
    feature_names : list[str]
        Feature names.
    n_numeric_features : int
        Number of numeric feature columns.

    Returns
    -------
    dict[str, dict[str, list[dict] | dict[float, dict]]]
        Nested dict: model_name → perturbation_type → results.
    """
    all_results = {}

    for model_name in models:
        print(f"\n  === Perturbations for {model_name} ===")
        model_results = {}

        print("    [1/4] Seed variation")
        model_results["seed"] = run_seed_perturbation(
            model_name, X_train, y_train, X_test, local_indices, feature_names
        )

        print("    [2/4] Bootstrap resampling")
        model_results["bootstrap"] = run_bootstrap_perturbation(
            model_name, X_train, y_train, X_test, local_indices, feature_names
        )

        print("    [3/4] Gaussian noise")
        model_results["noise"] = run_noise_perturbation(
            model_name,
            X_train,
            y_train,
            X_test,
            local_indices,
            feature_names,
            n_numeric_features,
        )

        print("    [4/4] Dataset size variation")
        model_results["size"] = run_size_perturbation(
            model_name, X_train, y_train, X_test, local_indices, feature_names
        )

        all_results[model_name] = model_results

    print("\n✓ All perturbation experiments complete.")
    return all_results
