"""
features.py — Engineer the full feature set for the AISD prediction model.

Input : data/raw/panel.parquet  (output of merge.py)
Output: data/raw/features.parquet

Feature groups
--------------
Borrower (from Compustat):
    leverage, profitability, interest_coverage, log_firm_size,
    tangibility, cash_ratio, altman_z (and its five components)

Loan structural (from DealScan):
    log_facility_amount, maturity_months, collateral_flag,
    covenant_count, is_revolver, loan_purpose dummies

Market / macro:
    eq_vol_12m (from CRSP), baa_aaa_spread, risk_free_10y (from FRED)

Usage:
    python src/features.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW  = ROOT / "data" / "raw"

PANEL_IN   = RAW / "panel.parquet"
FEAT_OUT   = RAW / "features.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# Minimum total assets to avoid division noise
MIN_AT = 10.0   # $10 million

# Last origination date in the training split. Winsorisation bounds are
# estimated on training-period loans only, then applied to every split, so no
# validation/test information leaks into the features. model.py imports this.
TRAIN_END = "2016-12-31"


# ---------------------------------------------------------------------------
# Borrower financial ratios
# ---------------------------------------------------------------------------

def add_borrower_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute balance-sheet and income-statement ratios from Compustat funda.

    Winsorises all ratios at the 1st/99th percentile of the training period
    (origination_date <= TRAIN_END) to limit outlier influence.

    Parameters
    ----------
    df : DataFrame with Compustat columns:
         total_assets, long_term_debt, short_term_debt, ebitda,
         interest_expense, net_ppe, cash_and_equiv, book_equity,
         net_income, working_capital, retained_earnings, sales.
    """
    df = df.copy()

    at   = df["total_assets"].clip(lower=MIN_AT)
    dltt = df["long_term_debt"].fillna(0)
    dlc  = df["short_term_debt"].fillna(0)

    # Core ratios
    df["leverage"]          = (dltt + dlc) / at
    df["profitability"]     = df["ebitda"].fillna(0) / at
    df["interest_coverage"] = np.where(
        df["interest_expense"].fillna(0).abs() > 0.01,
        df["ebitda"].fillna(0) / df["interest_expense"].abs(),
        np.nan
    )
    df["log_firm_size"]     = np.log(at)
    df["tangibility"]       = df["net_ppe"].fillna(0) / at
    df["cash_ratio"]        = df["cash_and_equiv"].fillna(0) / at

    # Altman Z'-score components (Altman 1983; private-firm variant, since X4
    # uses book equity rather than market equity — appropriate here because
    # borrowers may be privately held or delisted).
    #   Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5
    #   Distress threshold: Z' < 1.23
    df["altman_x1"] = df["working_capital"].fillna(0) / at           # liquidity
    df["altman_x2"] = df["retained_earnings"].fillna(0) / at         # accumulated profitability
    df["altman_x3"] = df["ebitda"].fillna(0) / at                    # operating efficiency
    # X4 = book equity / total liabilities. Total liabilities is approximated
    # as total assets minus book equity (Compustat does not need a separate
    # `lt` pull for this); the approximation absorbs preferred stock and
    # minority interest, which are immaterial for most industrial borrowers.
    book_equity = df["book_equity"].fillna(0)
    total_liabilities = (at - book_equity).clip(lower=0.001)
    df["altman_x4"] = book_equity / total_liabilities                # financial structure
    df["altman_x5"] = df["sales"].fillna(0) / at                     # asset turnover

    df["altman_z"] = (
        0.717 * df["altman_x1"] +
        0.847 * df["altman_x2"] +
        3.107 * df["altman_x3"] +
        0.420 * df["altman_x4"] +
        0.998 * df["altman_x5"]
    )

    # Winsorise all float features at 1/99, with bounds from training loans only
    ratio_cols = [
        "leverage", "profitability", "interest_coverage",
        "tangibility", "cash_ratio",
        "altman_x1", "altman_x2", "altman_x3", "altman_x4", "altman_x5",
        "altman_z"
    ]
    train = df["origination_date"] <= TRAIN_END
    for col in ratio_cols:
        lo, hi = df.loc[train, col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    log.info("Borrower features added.")
    return df


# ---------------------------------------------------------------------------
# Loan structural features
# ---------------------------------------------------------------------------

def add_loan_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute loan-level features from DealScan facility data.

    Generates:
        log_facility_amount : log of facility_amount_mm
        maturity_months     : time from origination to maturity in months
        loan_purpose_*      : one-hot dummies for primary loan purpose
        (collateral_flag and is_revolver are passed through as-is)

    Parameters
    ----------
    df : DataFrame with DealScan columns:
         facility_amount_mm, origination_date, maturity,
         loan_purpose, collateral_flag, is_revolver.
    """
    df = df.copy()

    df["log_facility_amount"] = np.log(
        df["facility_amount_mm"].clip(lower=0.1)
    )

    # Maturity in months
    if "maturity" in df.columns:
        df["maturity"] = pd.to_datetime(df["maturity"], errors="coerce")
        df["maturity_months"] = (
            (df["maturity"] - df["origination_date"])
            / np.timedelta64(1, "D") / 30.44
        ).round(0)
    else:
        df["maturity_months"] = np.nan

    df["maturity_months"] = df["maturity_months"].clip(lower=1, upper=360)

    # Loan purpose dummies (consolidate rare categories into "other")
    top_purposes = [
        "Corp. purposes", "Debt repay", "Acquisition line",
        "LBO", "Working capital", "Recapitalization"
    ]
    df["loan_purpose_clean"] = np.where(
        df["loan_purpose"].isin(top_purposes),
        df["loan_purpose"].str.replace(r"\s+", "_", regex=True).str.lower(),
        "other"
    )
    purpose_dummies = pd.get_dummies(
        df["loan_purpose_clean"], prefix="purpose", drop_first=False
    )
    df = pd.concat([df, purpose_dummies], axis=1)

    # Ensure binary flags are 0/1 integers
    df["collateral_flag"] = df["collateral_flag"].fillna(0).astype(int)
    df["is_revolver"]     = df["is_revolver"].fillna(0).astype(int)

    log.info("Loan structural features added.")
    return df


# ---------------------------------------------------------------------------
# Market / macro features
# ---------------------------------------------------------------------------

def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate market and macro features already in the panel.

    Columns expected (from merge.py):
        eq_vol_12m     — rolling 12-month equity return volatility
        baa_aaa_spread — BAA minus AAA yield in bps
        risk_free_10y  — 10-year Treasury yield (%)

    Winsorises eq_vol_12m at the 99th percentile of the training period.
    """
    df = df.copy()

    if "eq_vol_12m" in df.columns:
        train = df["origination_date"] <= TRAIN_END
        hi = df.loc[train, "eq_vol_12m"].quantile(0.99)
        df["eq_vol_12m"] = df["eq_vol_12m"].clip(upper=hi)

    log.info("Market/macro features validated.")
    return df


# ---------------------------------------------------------------------------
# Target variable
# ---------------------------------------------------------------------------

def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the target variable (aisd_bps) is clean and within a
    reasonable range.  Rows with missing AISD are dropped.

    Parameters
    ----------
    df : DataFrame with column aisd_bps.
    """
    before = len(df)
    df = df.dropna(subset=["aisd_bps"])
    df = df[df["aisd_bps"].between(0, 2000)]
    log.info("Target cleaned: dropped %d rows with invalid AISD; %d remain",
             before - len(df), len(df))
    return df


# ---------------------------------------------------------------------------
# Full feature build
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Borrower
    "leverage", "profitability", "interest_coverage",
    "log_firm_size", "tangibility", "cash_ratio",
    "altman_x1", "altman_x2", "altman_x3", "altman_x4", "altman_x5",
    "altman_z",
    # Loan
    "log_facility_amount", "maturity_months",
    "collateral_flag", "covenant_count", "is_revolver",
    # Market / macro
    "eq_vol_12m", "baa_aaa_spread", "risk_free_10y",
]


def build_features() -> pd.DataFrame:
    """
    Load the raw panel, engineer all features, and write the feature matrix.

    Returns
    -------
    pd.DataFrame with engineered features, target column (aisd_bps),
    and metadata columns (facilityid, gvkey, origination_date).
    """
    panel = pd.read_parquet(PANEL_IN)
    log.info("Panel loaded: %d rows", len(panel))

    panel = prepare_target(panel)
    panel = add_borrower_features(panel)
    panel = add_loan_features(panel)
    panel = add_market_features(panel)

    # Collect purpose dummy column names dynamically
    purpose_dummies = [c for c in panel.columns if c.startswith("purpose_")]
    all_features = FEATURE_COLS + purpose_dummies

    # Keep only columns that exist in the DataFrame
    available = [c for c in all_features if c in panel.columns]
    missing   = set(all_features) - set(available)
    if missing:
        log.warning("Missing feature columns (will be excluded): %s", missing)

    out_cols = ["facilityid", "gvkey", "origination_date", "aisd_bps"] + available
    out_cols = [c for c in out_cols if c in panel.columns]
    result = panel[out_cols].copy()

    # Log missing-value summary
    na_pct = result[available].isna().mean().sort_values(ascending=False)
    log.info("Top-5 features by missing rate:\n%s", na_pct.head())

    result.to_parquet(FEAT_OUT, index=False)
    log.info("Feature matrix written to %s — %d rows, %d features",
             FEAT_OUT, len(result), len(available))
    return result


if __name__ == "__main__":
    build_features()
