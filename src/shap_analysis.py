"""
shap_analysis.py - SHAP explainability for the XGBoost AISD model.

Produces:
    1. SHAP summary plot (beeswarm) -- feature importance + direction
    2. SHAP bar chart -- mean absolute SHAP values
    3. Partial dependence plots for leverage and log_firm_size

All plots are saved to data/raw/figures/.

Usage:
    python src/shap_analysis.py
"""

import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FEAT_IN = RAW / "features.parquet"
MODELS_DIR = RAW / "models"
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load artefacts
# ---------------------------------------------------------------------------

def load_artefacts():
    """
    Load the trained XGBoost model, scaler metadata, and feature matrix.

    Returns
    -------
    Tuple of (xgb_model, X_test_sc, feature_cols, X_test_df).
    """
    # Model
    xgb = XGBRegressor()
    xgb.load_model(str(MODELS_DIR / "xgboost.ubj"))

    # Scaler / feature metadata
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        meta = pickle.load(f)
    scaler = meta["scaler"]
    medians = meta["medians"]
    feature_cols = meta["feature_cols"]

    # Feature matrix -- use the test window (2020-2022) for SHAP
    df = pd.read_parquet(FEAT_IN)
    df["origination_date"] = pd.to_datetime(df["origination_date"])
    test = df[df["origination_date"] > "2019-12-31"].copy()

    X_test = test[feature_cols].fillna(medians)
    X_test_sc = scaler.transform(X_test)

    return xgb, X_test_sc, feature_cols, X_test


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(
    xgb: XGBRegressor,
    X_sc: np.ndarray,
) -> shap.Explanation:
    """
    Compute SHAP values using TreeExplainer (exact, model-native).

    Parameters
    ----------
    xgb  : Fitted XGBRegressor.
    X_sc : Scaled feature matrix (numpy array).

    Returns
    -------
    shap.Explanation object.
    """
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer(X_sc)
    log.info("SHAP values computed for %d observations.", X_sc.shape[0])
    return shap_values


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_summary_beeswarm(
    shap_values: shap.Explanation,
    feature_cols: list,
    max_display: int = 20,
) -> None:
    """
    Save a SHAP beeswarm summary plot (top-N features by mean |SHAP|).

    Parameters
    ----------
    shap_values   : SHAP Explanation object.
    feature_cols  : Feature names in the same order as the model input.
    max_display   : Number of features to display.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        feature_names=feature_cols,
        max_display=max_display,
        show=False,
    )
    plt.title("SHAP Summary -- XGBoost AISD Model", fontsize=13, pad=12)
    plt.tight_layout()
    out = FIGURES_DIR / "shap_summary_beeswarm.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_summary_bar(
    shap_values: shap.Explanation,
    feature_cols: list,
    max_display: int = 20,
) -> None:
    """
    Save a SHAP bar chart of mean absolute SHAP values.

    Parameters
    ----------
    shap_values  : SHAP Explanation object.
    feature_cols : Feature names.
    max_display  : Number of features to show.
    """
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    names = [feature_cols[i] for i in order]
    vals = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(len(names)), vals[::-1], color="#4c72b0")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (bps)", fontsize=10)
    ax.set_title("Feature Importance -- Mean |SHAP|", fontsize=12)
    plt.tight_layout()
    out = FIGURES_DIR / "shap_summary_bar.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


def plot_partial_dependence(
    shap_values: shap.Explanation,
    X_df: pd.DataFrame,
    feature: str,
    feature_cols: list,
) -> None:
    """
    Save a SHAP partial dependence (dependence) plot for one feature.

    The colour axis shows the feature with the highest interaction effect.

    Parameters
    ----------
    shap_values  : SHAP Explanation object.
    X_df         : Original (unscaled) feature DataFrame for axis labels.
    feature      : Name of the feature to plot on the x-axis.
    feature_cols : List of all feature names.
    """
    if feature not in feature_cols:
        log.warning("Feature '%s' not found in model; skipping.", feature)
        return

    feat_idx = feature_cols.index(feature)
    fig, ax = plt.subplots(figsize=(7, 5))
    shap.dependence_plot(
        feat_idx,
        shap_values.values,
        X_df[feature_cols].values,
        feature_names=feature_cols,
        ax=ax,
        show=False,
    )
    ax.set_xlabel(feature, fontsize=10)
    ax.set_ylabel(f"SHAP value for {feature} (bps)", fontsize=10)
    ax.set_title(f"Partial Dependence: {feature}", fontsize=12)
    plt.tight_layout()
    safe_name = feature.replace("/", "_")
    out = FIGURES_DIR / f"shap_pdp_{safe_name}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_shap_analysis() -> None:
    """
    Orchestrate SHAP analysis: load artefacts, compute SHAP, save all plots.
    """
    xgb, X_test_sc, feature_cols, X_test_df = load_artefacts()
    shap_values = compute_shap_values(xgb, X_test_sc)

    plot_summary_beeswarm(shap_values, feature_cols)
    plot_summary_bar(shap_values, feature_cols)
    plot_partial_dependence(shap_values, X_test_df, "leverage", feature_cols)
    plot_partial_dependence(
        shap_values, X_test_df, "log_firm_size", feature_cols
    )
    log.info("SHAP analysis complete. Figures in %s", FIGURES_DIR)


if __name__ == "__main__":
    run_shap_analysis()
