"""
model.py - Train and evaluate AISD spread prediction models.

Split strategy (time-aware, no data leakage):
    Train   : 2010-2016
    Validate: 2017-2019
    Test    : 2020-2022

Models:
    1. OLS baseline (statsmodels, HC3 robust SEs)
    2. Lasso (sklearn, alpha tuned on validation set)
    3. Ridge (sklearn, alpha tuned on validation set)
    4. XGBoost (early stopping on validation set)

Evaluation metrics: RMSE, MAE, R2 out-of-sample on the test set.
All fitted objects saved to data/raw/models/ for downstream SHAP analysis.

Usage:
    python src/model.py
"""

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from features import TRAIN_END  # single source for the train/validate boundary

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

FEAT_IN = RAW / "features.parquet"
MODELS_DIR = RAW / "models"

VALIDATE_END = "2019-12-31"
TEST_END = "2022-12-31"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------

def time_split(df: pd.DataFrame):
    """
    Split feature matrix into train/validate/test by origination_date.

    Parameters
    ----------
    df : DataFrame with [origination_date, aisd_bps, <feature columns>].

    Returns
    -------
    Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols).
    """
    df = df.copy()
    df["origination_date"] = pd.to_datetime(df["origination_date"])

    meta_cols = {"facilityid", "cik", "gvkey", "origination_date", "aisd_bps"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    train = df[df["origination_date"] <= TRAIN_END]
    val = df[
        (df["origination_date"] > TRAIN_END)
        & (df["origination_date"] <= VALIDATE_END)
    ]
    test = df[
        (df["origination_date"] > VALIDATE_END)
        & (df["origination_date"] <= TEST_END)
    ]

    log.info(
        "Split sizes -- train: %d, val: %d, test: %d",
        len(train), len(val), len(test),
    )

    def xy(split):
        return split[feature_cols].copy(), split["aisd_bps"].values

    X_tr, y_tr = xy(train)
    X_vl, y_vl = xy(val)
    X_te, y_te = xy(test)
    return X_tr, X_vl, X_te, y_tr, y_vl, y_te, feature_cols


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def impute_and_scale(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Median-impute missing values and z-score scale using train-set statistics.

    All statistics are computed on the training split only to prevent leakage.

    Returns
    -------
    Tuple of (X_train_sc, X_val_sc, X_test_sc, scaler, medians).
    """
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)
    X_test_sc = scaler.transform(X_test)
    return X_train_sc, X_val_sc, X_test_sc, scaler, medians


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    """
    Compute RMSE, MAE, and R2 for a set of predictions.

    Parameters
    ----------
    y_true : Ground-truth AISD values (basis points).
    y_pred : Model predictions.
    name   : Label used in the comparison table.

    Returns
    -------
    dict with keys {model, rmse, mae, r2}.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "model": name,
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
    }


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_ols(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list,
):
    """
    Fit OLS baseline via statsmodels with HC3 heteroskedasticity-robust SEs.

    Parameters
    ----------
    X_train      : Scaled training feature matrix (numpy array).
    y_train      : Training target vector.
    feature_cols : Feature names for printing the coefficient table.
    """
    X_const = sm.add_constant(X_train)
    ols = sm.OLS(y_train, X_const).fit(cov_type="HC3")
    log.info("OLS in-sample R2: %.4f", ols.rsquared)
    return ols


def tune_regularised(Model, X_tr, y_tr, X_vl, y_vl, alphas, **kwargs):
    """
    Grid-search regularisation strength on the validation set.

    Parameters
    ----------
    Model  : sklearn estimator class (Lasso or Ridge).
    alphas : Iterable of regularisation strengths to evaluate.
    kwargs : Additional keyword arguments forwarded to Model().

    Returns
    -------
    Tuple of (best_fitted_model, best_alpha).
    """
    best_rmse = np.inf
    best_alpha = None
    best_model = None
    for alpha in alphas:
        m = Model(alpha=alpha, max_iter=10_000, **kwargs)
        m.fit(X_tr, y_tr)
        rmse = np.sqrt(mean_squared_error(y_vl, m.predict(X_vl)))
        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = alpha
            best_model = m
    log.info(
        "%s best alpha=%.4f (val RMSE=%.2f bps)",
        Model.__name__, best_alpha, best_rmse,
    )
    return best_model, best_alpha


def train_xgboost(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_vl: np.ndarray,
    y_vl: np.ndarray,
) -> XGBRegressor:
    """
    Train XGBoost with early stopping monitored on the validation set.

    The validation set is used solely for early stopping, not for
    hyperparameter search, so it does not constitute data leakage.

    Parameters
    ----------
    X_tr, y_tr : Training features and targets.
    X_vl, y_vl : Validation features and targets.
    """
    xgb = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric="rmse",
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_vl, y_vl)],
        verbose=100,
    )
    log.info("XGBoost best iteration: %d", xgb.best_iteration)
    return xgb


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_models() -> pd.DataFrame:
    """
    Load the feature matrix, train all models, and print a comparison table.

    Saves each fitted model to data/raw/models/ for use in shap_analysis.py.

    Returns
    -------
    pd.DataFrame -- comparison table (models x metrics).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(FEAT_IN)
    log.info("Feature matrix loaded: %d rows, %d columns", *df.shape)

    X_tr, X_vl, X_te, y_tr, y_vl, y_te, fcols = time_split(df)
    X_tr_sc, X_vl_sc, X_te_sc, scaler, medians = impute_and_scale(
        X_tr, X_vl, X_te
    )

    results = []
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 500.0]

    # OLS
    ols = train_ols(X_tr_sc, y_tr, fcols)
    X_te_const = sm.add_constant(X_te_sc, has_constant="add")
    results.append(evaluate(y_te, ols.predict(X_te_const), "OLS"))
    with open(MODELS_DIR / "ols.pkl", "wb") as f:
        pickle.dump(ols, f)

    # Lasso
    lasso, _ = tune_regularised(Lasso, X_tr_sc, y_tr, X_vl_sc, y_vl, alphas)
    results.append(evaluate(y_te, lasso.predict(X_te_sc), "Lasso"))
    with open(MODELS_DIR / "lasso.pkl", "wb") as f:
        pickle.dump(lasso, f)

    # Ridge
    ridge, _ = tune_regularised(Ridge, X_tr_sc, y_tr, X_vl_sc, y_vl, alphas)
    results.append(evaluate(y_te, ridge.predict(X_te_sc), "Ridge"))
    with open(MODELS_DIR / "ridge.pkl", "wb") as f:
        pickle.dump(ridge, f)

    # XGBoost
    xgb = train_xgboost(X_tr_sc, y_tr, X_vl_sc, y_vl)
    results.append(evaluate(y_te, xgb.predict(X_te_sc), "XGBoost"))
    xgb.save_model(str(MODELS_DIR / "xgboost.ubj"))

    # Persist scaler and feature metadata for inference scripts
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(
            {"scaler": scaler, "medians": medians, "feature_cols": fcols}, f
        )

    comparison = pd.DataFrame(results).set_index("model")
    print("\n=== Out-of-sample model comparison (test 2020-2022) ===")
    print(comparison.to_string())
    comparison.to_csv(RAW / "model_comparison.csv")
    return comparison


if __name__ == "__main__":
    run_models()
