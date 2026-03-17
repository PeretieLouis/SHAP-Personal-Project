"""Stability metrics for SHAP robustness evaluation.

Computes quantitative metrics that compare perturbed SHAP explanations
against the baseline to assess explanation stability.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def _global_importance_vector(result: dict) -> np.ndarray:
    """Extract the mean |SHAP| vector in original feature order."""
    imp = result["global_importance"]
    # Undo the sort: reconstruct mean_abs in original column order
    n_features = len(imp["ranking"])
    vec = np.empty(n_features)
    vec[imp["ranking"]] = imp["mean_abs_shap"]
    return vec


def _rank_vector(importance_vec: np.ndarray) -> np.ndarray:
    """Convert importance vector to rank (1 = most important)."""
    # argsort of descending importance gives rank positions
    order = np.argsort(-importance_vec)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# ──────────────────────────────────────────────
# Metric 1: Spearman rank correlation
# ──────────────────────────────────────────────
def spearman_rank_correlation(
    baseline_result: dict,
    perturbed_results: list[dict],
) -> list[float]:
    """Spearman correlation between baseline and perturbed feature rankings.

    Parameters
    ----------
    baseline_result : dict
        Baseline SHAP result (from compute_baseline_shap).
    perturbed_results : list[dict]
        List of perturbation result dicts.

    Returns
    -------
    list[float]
        One Spearman ρ per perturbation run.
    """
    base_vec = _global_importance_vector(baseline_result)
    base_rank = _rank_vector(base_vec)

    correlations = []
    for res in perturbed_results:
        pert_vec = _global_importance_vector(res)
        pert_rank = _rank_vector(pert_vec)
        rho, _ = spearmanr(base_rank, pert_rank)
        correlations.append(rho)
    return correlations


# ──────────────────────────────────────────────
# Metric 2: Cosine similarity
# ──────────────────────────────────────────────
def cosine_similarity(
    baseline_result: dict,
    perturbed_results: list[dict],
) -> list[float]:
    """Cosine similarity between baseline and perturbed global importance.

    Parameters
    ----------
    baseline_result : dict
        Baseline SHAP result.
    perturbed_results : list[dict]
        List of perturbation result dicts.

    Returns
    -------
    list[float]
        One cosine similarity per perturbation run.
    """
    base_vec = _global_importance_vector(baseline_result)
    similarities = []
    for res in perturbed_results:
        pert_vec = _global_importance_vector(res)
        # scipy.spatial.distance.cosine returns *distance*, so 1 - dist = similarity
        sim = 1.0 - cosine(base_vec, pert_vec)
        similarities.append(sim)
    return similarities


# ──────────────────────────────────────────────
# Metric 3: SHAP variance per feature
# ──────────────────────────────────────────────
def shap_variance_per_feature(
    perturbed_results: list[dict],
    feature_names: list[str],
) -> pd.DataFrame:
    """Variance of mean |SHAP| across perturbation runs, per feature.

    Parameters
    ----------
    perturbed_results : list[dict]
        List of perturbation result dicts.
    feature_names : list[str]
        Feature names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [feature, shap_variance], sorted descending.
    """
    importance_matrix = np.array(
        [_global_importance_vector(r) for r in perturbed_results]
    )  # shape: (n_runs, n_features)

    variance = importance_matrix.var(axis=0)

    df = pd.DataFrame({"feature": feature_names, "shap_variance": variance})
    return df.sort_values("shap_variance", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────
# Metric 4: Stability Index S
# ──────────────────────────────────────────────
def stability_index(
    baseline_result: dict,
    perturbed_results: list[dict],
) -> float:
    """Aggregate stability index combining rank correlation and cosine similarity.

    S = 0.5 × mean(Spearman ρ) + 0.5 × mean(cosine similarity)

    A score of 1.0 means perfect stability; lower is less stable.

    Parameters
    ----------
    baseline_result : dict
        Baseline SHAP result.
    perturbed_results : list[dict]
        List of perturbation result dicts.

    Returns
    -------
    float
        Stability index in [0, 1].
    """
    spearman_vals = spearman_rank_correlation(baseline_result, perturbed_results)
    cosine_vals = cosine_similarity(baseline_result, perturbed_results)
    return 0.5 * np.mean(spearman_vals) + 0.5 * np.mean(cosine_vals)


# ──────────────────────────────────────────────
# Aggregation across models and perturbation types
# ──────────────────────────────────────────────
def compute_all_metrics(
    baseline_shap: dict[str, dict],
    perturbation_results: dict[str, dict],
    feature_names: list[str],
) -> dict[str, pd.DataFrame]:
    """Compute all stability metrics across models and perturbation types.

    Parameters
    ----------
    baseline_shap : dict[str, dict]
        Per-model baseline SHAP results from compute_baseline_shap.
    perturbation_results : dict[str, dict[str, list[dict] | dict]]
        Nested dict: model_name → perturbation_type → results.
        For 'size' perturbation, the value is a dict[float, dict].
    feature_names : list[str]
        Feature names.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys: 'spearman', 'cosine', 'shap_variance', 'stability_index'.
    """
    # Perturbation types that produce list[dict]
    list_perturbation_types = ["seed", "bootstrap", "noise"]

    spearman_rows = []
    cosine_rows = []
    variance_frames = []
    si_rows = []

    for model_name, pert_dict in perturbation_results.items():
        baseline = baseline_shap[model_name]

        for pert_type in list_perturbation_types:
            results_list = pert_dict[pert_type]

            # Spearman
            sp_vals = spearman_rank_correlation(baseline, results_list)
            for i, val in enumerate(sp_vals):
                spearman_rows.append(
                    {
                        "model": model_name,
                        "perturbation": pert_type,
                        "run": i,
                        "spearman_rho": val,
                    }
                )

            # Cosine
            cos_vals = cosine_similarity(baseline, results_list)
            for i, val in enumerate(cos_vals):
                cosine_rows.append(
                    {
                        "model": model_name,
                        "perturbation": pert_type,
                        "run": i,
                        "cosine_similarity": val,
                    }
                )

            # SHAP variance
            var_df = shap_variance_per_feature(results_list, feature_names)
            var_df["model"] = model_name
            var_df["perturbation"] = pert_type
            variance_frames.append(var_df)

            # Stability index
            si = stability_index(baseline, results_list)
            si_rows.append(
                {
                    "model": model_name,
                    "perturbation": pert_type,
                    "stability_index": si,
                }
            )

        # Handle size perturbation (dict of fraction → single result)
        size_results = pert_dict["size"]
        size_list = list(size_results.values())
        size_fracs = list(size_results.keys())

        sp_vals = spearman_rank_correlation(baseline, size_list)
        cos_vals = cosine_similarity(baseline, size_list)
        for frac, sp, cs in zip(size_fracs, sp_vals, cos_vals, strict=True):
            spearman_rows.append(
                {
                    "model": model_name,
                    "perturbation": f"size_{frac:.0%}",
                    "run": 0,
                    "spearman_rho": sp,
                }
            )
            cosine_rows.append(
                {
                    "model": model_name,
                    "perturbation": f"size_{frac:.0%}",
                    "run": 0,
                    "cosine_similarity": cs,
                }
            )

        # For size perturbation, variance is across fractions
        if len(size_list) > 1:
            var_df = shap_variance_per_feature(size_list, feature_names)
            var_df["model"] = model_name
            var_df["perturbation"] = "size"
            variance_frames.append(var_df)

        si = stability_index(baseline, size_list)
        si_rows.append(
            {
                "model": model_name,
                "perturbation": "size",
                "stability_index": si,
            }
        )

    return {
        "spearman": pd.DataFrame(spearman_rows),
        "cosine": pd.DataFrame(cosine_rows),
        "shap_variance": pd.concat(variance_frames, ignore_index=True),
        "stability_index": pd.DataFrame(si_rows),
    }


def save_metrics(
    metrics: dict[str, pd.DataFrame],
    output_dir,
) -> None:
    """Save all metric DataFrames to CSV.

    Parameters
    ----------
    metrics : dict[str, pd.DataFrame]
        Output of compute_all_metrics.
    output_dir : Path
        Directory to write CSV files into.
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, df in metrics.items():
        path = output_dir / f"robustness_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {path}")
