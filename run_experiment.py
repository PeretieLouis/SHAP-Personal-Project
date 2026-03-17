"""Single-entrypoint script that runs the full experiment pipeline.

Usage:
    uv run python run_experiment.py

Steps executed:
    1. Load raw data
    2. Preprocess (split, transform, save)
    3. Train all models
    4. Evaluate models
    5. Compute baseline SHAP explanations + plots
    6. Run robustness perturbation experiments
    7. Compute stability metrics + save CSVs
    8. Generate robustness plots
"""

from src.config import METRICS_DIR, N_LOCAL_SAMPLES, NUMERIC_FEATURES
from src.data.load_data import load_raw_data
from src.data.preprocess import run_preprocessing_pipeline
from src.explainability.compute_shap import compute_baseline_shap
from src.explainability.shap_analysis import generate_baseline_plots
from src.models.evaluate_models import (
    evaluate_all_models,
    print_classification_reports,
    save_evaluation_results,
)
from src.models.train_models import train_all_models
from src.robustness.perturbations import run_all_perturbations
from src.robustness.stability_metrics import compute_all_metrics, save_metrics
from src.visualization.plots import generate_robustness_plots


def main() -> None:
    # ── Step 1: Load data ─────────────────────────
    print("=" * 60)
    print("STEP 1 — Loading raw data")
    print("=" * 60)
    df = load_raw_data()
    print(f"  Loaded {len(df):,} rows, {df.shape[1]} columns")

    # ── Step 2: Preprocess ────────────────────────
    print("\n" + "=" * 60)
    print("STEP 2 — Preprocessing")
    print("=" * 60)
    X_train, X_test, y_train, y_test, feature_names, _ = run_preprocessing_pipeline(df)

    # ── Step 3: Train models ──────────────────────
    print("\n" + "=" * 60)
    print("STEP 3 — Training models")
    print("=" * 60)
    models = train_all_models(X_train, y_train.values)

    # ── Step 4: Evaluate models ───────────────────
    print("\n" + "=" * 60)
    print("STEP 4 — Evaluating models")
    print("=" * 60)
    df_eval = evaluate_all_models(models, X_test, y_test.values)
    save_evaluation_results(df_eval)
    print_classification_reports(models, X_test, y_test.values)

    # ── Step 5: Baseline SHAP ─────────────────────
    print("\n" + "=" * 60)
    print("STEP 5 — Computing baseline SHAP explanations")
    print("=" * 60)
    baseline_shap = compute_baseline_shap(
        models, X_train, X_test, feature_names, n_local_samples=N_LOCAL_SAMPLES
    )
    generate_baseline_plots(baseline_shap, X_test, feature_names)

    # ── Step 6: Robustness perturbations ──────────
    print("\n" + "=" * 60)
    print("STEP 6 — Running perturbation experiments")
    print("=" * 60)
    local_indices = baseline_shap[next(iter(baseline_shap))]["local_indices"]
    perturbation_results = run_all_perturbations(
        models,
        X_train,
        y_train.values,
        X_test,
        local_indices,
        feature_names,
        n_numeric_features=len(NUMERIC_FEATURES),
    )

    # ── Step 7: Stability metrics ─────────────────
    print("\n" + "=" * 60)
    print("STEP 7 — Computing stability metrics")
    print("=" * 60)
    metrics = compute_all_metrics(baseline_shap, perturbation_results, feature_names)
    save_metrics(metrics, METRICS_DIR)

    # Print summary
    print("\n  Stability Index summary:")
    print(metrics["stability_index"].to_string(index=False))

    # ── Step 8: Robustness plots ──────────────────
    print("\n" + "=" * 60)
    print("STEP 8 — Generating robustness plots")
    print("=" * 60)
    generate_robustness_plots(baseline_shap, perturbation_results, metrics, feature_names)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
