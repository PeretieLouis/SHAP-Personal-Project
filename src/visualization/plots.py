"""Robustness experiment visualizations.

Generates plots for SHAP stability analysis across perturbation types
and models. All figures are saved to outputs/plots/.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PLOTS_DIR

matplotlib.use("Agg")


def _ensure_plots_dir() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# Plot 1: Feature importance ranking changes
# ──────────────────────────────────────────────
def plot_ranking_changes(
    baseline_shap: dict[str, dict],
    perturbation_results: dict[str, dict],
    feature_names: list[str],
    top_k: int = 15,
) -> None:
    """Show how feature importance ranks shift across perturbation runs.

    Plots baseline ranking vs. range of perturbed rankings for each model
    and perturbation type.

    Parameters
    ----------
    baseline_shap : dict[str, dict]
        Per-model baseline SHAP results.
    perturbation_results : dict[str, dict[str, list[dict]]]
        Nested: model → perturbation_type → list of result dicts.
    feature_names : list[str]
        Feature names.
    top_k : int
        Number of top features to display.
    """
    from src.robustness.stability_metrics import (
        _global_importance_vector,
        _rank_vector,
    )

    _ensure_plots_dir()
    pert_types = ["seed", "bootstrap", "noise"]

    for model_name in perturbation_results:
        baseline = baseline_shap[model_name]
        base_imp = _global_importance_vector(baseline)
        base_rank = _rank_vector(base_imp)

        # Select top-k features by baseline importance
        top_idx = np.argsort(base_rank)[:top_k]
        top_names = [feature_names[i] for i in top_idx]

        for pert_type in pert_types:
            results_list = perturbation_results[model_name][pert_type]

            # Collect ranks across runs
            pert_ranks = []
            for res in results_list:
                vec = _global_importance_vector(res)
                pert_ranks.append(_rank_vector(vec))
            pert_ranks = np.array(pert_ranks)  # (n_runs, n_features)

            fig, ax = plt.subplots(figsize=(10, 6))
            y_pos = np.arange(top_k)

            # Baseline ranks
            base_vals = base_rank[top_idx]
            ax.barh(y_pos, base_vals, height=0.4, label="Baseline", alpha=0.7)

            # Perturbation range
            pert_min = pert_ranks[:, top_idx].min(axis=0)
            pert_max = pert_ranks[:, top_idx].max(axis=0)
            pert_mean = pert_ranks[:, top_idx].mean(axis=0)
            ax.errorbar(
                pert_mean,
                y_pos,
                xerr=[pert_mean - pert_min, pert_max - pert_mean],
                fmt="o",
                color="red",
                capsize=4,
                label="Perturbed (range)",
            )

            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_names, fontsize=8)
            ax.set_xlabel("Rank (1 = most important)")
            ax.set_title(f"Ranking Stability — {model_name} — {pert_type}", fontsize=13)
            ax.legend(loc="lower right")
            ax.invert_yaxis()
            plt.tight_layout()
            plt.savefig(
                PLOTS_DIR / f"ranking_changes_{model_name}_{pert_type}.png",
                dpi=150,
            )
            plt.close()

    print("  Saved ranking change plots.")


# ──────────────────────────────────────────────
# Plot 2: Spearman correlation distributions
# ──────────────────────────────────────────────
def plot_spearman_distributions(
    spearman_df: pd.DataFrame,
) -> None:
    """Box plot of Spearman ρ distributions per model and perturbation type.

    Parameters
    ----------
    spearman_df : pd.DataFrame
        DataFrame with columns [model, perturbation, run, spearman_rho].
    """
    _ensure_plots_dir()

    # Filter to list-type perturbations (skip size_* entries for cleaner viz)
    mask = spearman_df["perturbation"].isin(["seed", "bootstrap", "noise"])
    df = spearman_df[mask].copy()

    models = df["model"].unique()
    pert_types = df["perturbation"].unique()

    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models, strict=True):
        model_data = df[df["model"] == model]
        data_groups = [
            model_data[model_data["perturbation"] == pt]["spearman_rho"].values
            for pt in pert_types
        ]
        bp = ax.boxplot(data_groups, labels=pert_types, patch_artist=True)
        colors = ["#4C72B0", "#55A868", "#C44E52"]
        for patch, color in zip(bp["boxes"], colors, strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(model, fontsize=12)
        ax.set_ylabel("Spearman ρ" if ax == axes[0] else "")
        ax.set_ylim(0, 1.05)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Spearman Rank Correlation Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "spearman_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved Spearman distribution plot.")


# ──────────────────────────────────────────────
# Plot 3: SHAP variance per feature (top features)
# ──────────────────────────────────────────────
def plot_shap_variance(
    variance_df: pd.DataFrame,
    top_k: int = 15,
) -> None:
    """Bar chart of per-feature SHAP variance for each model/perturbation.

    Parameters
    ----------
    variance_df : pd.DataFrame
        DataFrame with columns [feature, shap_variance, model, perturbation].
    top_k : int
        Number of top features to show per subplot.
    """
    _ensure_plots_dir()

    models = variance_df["model"].unique()
    pert_types = variance_df["perturbation"].unique()

    for model in models:
        model_df = variance_df[variance_df["model"] == model]
        n_plots = len(pert_types)
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 6), sharey=False)
        if n_plots == 1:
            axes = [axes]

        for ax, pt in zip(axes, pert_types, strict=True):
            subset = (
                model_df[model_df["perturbation"] == pt]
                .nlargest(top_k, "shap_variance")
                .sort_values("shap_variance")
            )
            ax.barh(subset["feature"], subset["shap_variance"], color="#4C72B0")
            ax.set_xlabel("SHAP Variance")
            ax.set_title(pt, fontsize=11)
            ax.tick_params(axis="y", labelsize=8)

        fig.suptitle(f"SHAP Variance — {model}", fontsize=13)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"shap_variance_{model}.png", dpi=150)
        plt.close()

    print("  Saved SHAP variance plots.")


# ──────────────────────────────────────────────
# Plot 4: Stability Index comparison
# ──────────────────────────────────────────────
def plot_stability_index(
    si_df: pd.DataFrame,
) -> None:
    """Grouped bar chart comparing Stability Index across models and perturbations.

    Parameters
    ----------
    si_df : pd.DataFrame
        DataFrame with columns [model, perturbation, stability_index].
    """
    _ensure_plots_dir()

    models = si_df["model"].unique()
    pert_types = si_df["perturbation"].unique()
    n_pert = len(pert_types)

    x = np.arange(len(models))
    width = 0.8 / n_pert
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, pt in enumerate(pert_types):
        vals = [
            si_df[(si_df["model"] == m) & (si_df["perturbation"] == pt)]["stability_index"].values[
                0
            ]
            for m in models
        ]
        offset = (i - n_pert / 2 + 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=pt,
            color=colors[i % len(colors)],
            alpha=0.8,
        )
        for bar, val in zip(bars, vals, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Stability Index S")
    ax.set_ylim(0, 1.1)
    ax.set_title("Stability Index — Model × Perturbation Type", fontsize=14)
    ax.legend(title="Perturbation", loc="lower right")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "stability_index_comparison.png", dpi=150)
    plt.close()
    print("  Saved Stability Index comparison plot.")


# ──────────────────────────────────────────────
# Convenience: generate all robustness plots
# ──────────────────────────────────────────────
def generate_robustness_plots(
    baseline_shap: dict[str, dict],
    perturbation_results: dict[str, dict],
    metrics: dict[str, pd.DataFrame],
    feature_names: list[str],
) -> None:
    """Generate all robustness visualization plots.

    Parameters
    ----------
    baseline_shap : dict[str, dict]
        Baseline SHAP results from compute_baseline_shap.
    perturbation_results : dict[str, dict[str, list[dict] | dict]]
        Output of run_all_perturbations.
    metrics : dict[str, pd.DataFrame]
        Output of compute_all_metrics.
    feature_names : list[str]
        Feature names.
    """
    print("\n=== Generating Robustness Plots ===")
    plot_ranking_changes(baseline_shap, perturbation_results, feature_names)
    plot_spearman_distributions(metrics["spearman"])
    plot_shap_variance(metrics["shap_variance"])
    plot_stability_index(metrics["stability_index"])
    print("✓ All robustness plots saved.")
