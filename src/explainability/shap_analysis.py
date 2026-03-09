"""SHAP visualization and analysis utilities.

Generates summary plots, bar plots, and waterfall plots for
baseline SHAP explanations. All figures are saved to outputs/plots/.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import PLOTS_DIR

matplotlib.use("Agg")  # non-interactive backend for saving figures


def _ensure_plots_dir() -> None:
    """Create the plots output directory if it doesn't exist."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_summary(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list[str],
    model_name: str,
) -> None:
    """Generate and save a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values (n_samples × n_features).
    X : np.ndarray
        Feature matrix matching shap_values.
    feature_names : list[str]
        Feature names.
    model_name : str
        Used in the filename and title.
    """
    _ensure_plots_dir()
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=feature_names,
        show=False,
        max_display=20,
    )
    plt.title(f"SHAP Summary — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"shap_summary_{model_name}.png", dpi=150)
    plt.close()
    print(f"    Saved summary plot for {model_name}")


def plot_bar(
    shap_values: np.ndarray,
    feature_names: list[str],
    model_name: str,
    max_display: int = 20,
) -> None:
    """Generate and save a SHAP bar plot (mean |SHAP|).

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values (n_samples × n_features).
    feature_names : list[str]
        Feature names.
    model_name : str
        Used in the filename and title.
    max_display : int
        Number of top features to display.
    """
    _ensure_plots_dir()
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
        max_display=max_display,
    )
    plt.title(f"SHAP Feature Importance — {model_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"shap_bar_{model_name}.png", dpi=150)
    plt.close()
    print(f"    Saved bar plot for {model_name}")


def plot_waterfall(
    shap_values: np.ndarray,
    expected_value: float,
    X_sample: np.ndarray,
    feature_names: list[str],
    model_name: str,
    sample_idx: int,
) -> None:
    """Generate and save a SHAP waterfall plot for a single prediction.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for one sample (1-D array, n_features).
    expected_value : float
        Base value (expected model output).
    X_sample : np.ndarray
        Feature values for this sample (1-D array).
    feature_names : list[str]
        Feature names.
    model_name : str
        Used in the filename.
    sample_idx : int
        Index label for the sample (used in filename).
    """
    _ensure_plots_dir()

    explanation = shap.Explanation(
        values=shap_values,
        base_values=expected_value,
        data=X_sample,
        feature_names=feature_names,
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False, max_display=15)
    plt.title(f"Waterfall — {model_name} — Sample {sample_idx}", fontsize=12)
    plt.tight_layout()
    plt.savefig(
        PLOTS_DIR / f"shap_waterfall_{model_name}_sample{sample_idx}.png",
        dpi=150,
    )
    plt.close()


def _get_expected_value(explainer: shap.Explainer) -> float:
    """Extract scalar expected value (positive class) from an explainer.

    Parameters
    ----------
    explainer : shap.Explainer
        A fitted SHAP explainer.

    Returns
    -------
    float
        The base value for the positive class.
    """
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        return float(ev[1]) if len(ev) > 1 else float(ev[0])
    return float(ev)


def generate_baseline_plots(
    shap_results: dict[str, dict],
    X_test: np.ndarray,
    feature_names: list[str],
) -> None:
    """Generate all baseline SHAP plots for every model.

    Parameters
    ----------
    shap_results : dict[str, dict]
        Output of compute_baseline_shap().
    X_test : np.ndarray
        Full test feature matrix.
    feature_names : list[str]
        Feature names.
    """
    print("\n=== Generating SHAP Baseline Plots ===")
    for model_name, res in shap_results.items():
        print(f"  {model_name}:")

        # Summary (beeswarm) plot
        plot_summary(
            res["shap_values_test"],
            X_test,
            feature_names,
            model_name,
        )

        # Bar plot
        plot_bar(
            res["shap_values_test"],
            feature_names,
            model_name,
        )

        # Waterfall plots for local samples
        ev = _get_expected_value(res["explainer"])
        local_idx = res["local_indices"]
        for i, idx in enumerate(local_idx):
            plot_waterfall(
                shap_values=res["shap_values_local"][i],
                expected_value=ev,
                X_sample=X_test[idx],
                feature_names=feature_names,
                model_name=model_name,
                sample_idx=i,
            )
        print(f"    Saved {len(local_idx)} waterfall plots")

    print("✓ All baseline SHAP plots saved.")
