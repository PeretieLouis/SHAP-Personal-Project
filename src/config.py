"""Centralized experiment configuration.

All hyperparameters, seeds, paths, and experimental settings live here.
Nothing is hard-coded elsewhere in the codebase.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
METRICS_DIR = OUTPUTS_DIR / "metrics"

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.20

# ──────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────
TARGET_COLUMN = "income"
POSITIVE_LABEL = ">50K"

NUMERIC_FEATURES: list[str] = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES: list[str] = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]

# ──────────────────────────────────────────────
# Model hyperparameters
# ──────────────────────────────────────────────
LOGISTIC_REGRESSION_PARAMS: dict = {
    "C": 1.0,
    "max_iter": 1000,
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}

RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_leaf": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

XGBOOST_PARAMS: dict = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_SEED,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "verbosity": 0,
}

# ──────────────────────────────────────────────
# SHAP
# ──────────────────────────────────────────────
N_SHAP_BACKGROUND_SAMPLES = 100  # background samples for explainers
N_SHAP_EXPLAIN_SAMPLES = 500  # test samples to explain (global importance)
N_LOCAL_SAMPLES = 10  # test samples for local explanations

# ──────────────────────────────────────────────
# Robustness experiments
# ──────────────────────────────────────────────
N_PERTURBATION_RUNS = 3  # repeats per perturbation type
GAUSSIAN_NOISE_STD = 0.05  # std dev for Gaussian noise (relative to feature std)
BOOTSTRAP_SAMPLE_FRAC = 1.0  # fraction of training data per bootstrap
TRAINING_FRACTIONS: list[float] = [1.0, 0.6]  # dataset size variations

# Seeds for model random seed variation experiment
PERTURBATION_SEEDS: list[int] = list(range(100, 100 + N_PERTURBATION_RUNS))
