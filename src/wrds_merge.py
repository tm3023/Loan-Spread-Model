"""
wrds_merge.py — Merge WRDS data into a single panel for the loan spread model.

Replaces merge.py when using WRDS data instead of EDGAR scraping.

Pipeline:
    1. Load DealScan facilities (loan terms + AISD target).
    2. Join Chava-Roberts link on facilityid -> Compustat gvkey.
    3. As-of merge: each loan matched to most-recent Compustat fiscal year
       ending strictly before origination_date (no lookahead bias).
    4. Attach CRSP equity features via the CCM link (gvkey -> permno) and
       rolling 12-month return volatility from the month before origination.
    5. Attach FRED macro: BAA-AAA credit spread and 10-year Treasury yield.
    6. Write data/raw/panel.parquet.

Input files:
    data/raw/dealscan_facilities.csv   (from wrds_download.py)
    data/raw/chava_roberts_link.csv    (from convert_chava_roberts.py)
    data/raw/compustat_funda.csv       (from wrds_download.py)
    data/raw/crsp_monthly.csv          (from wrds_download.py)
    data/raw/crsp_ccm_link.csv         (from wrds_download.py)

Output: data/raw/panel.parquet

Usage:
    python src/wrds_merge.py
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PANEL_OUT = RAW / "panel.parquet"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COMPUSTAT_RENAME = {
    "at": "total_assets",
    "dltt": "long_term_debt",
    "dlc": "short_term_debt",
    "xint": "interest_expense",
    "ppent": "net_ppe",
    "che": "cash_and_equiv",
    "ceq": "book_equity",
    "ni": "net_income",
    "wcap": "working_capital",
    "re": "retained_earnings",
    "sale": "sales",
}

_PURPOSE_MAP = {
    "corporate purposes": "Corp. purposes",
    "general corporate": "Corp. purposes",
    "debt repayment": "Debt repay",
    "refinancing": "Debt repay",
    "acquisition": "Acquisition line",
    "acquisition line": "Acquisition line",
    "lbo": "LBO",
    "leveraged buyout": "LBO",
    "working capital": "Working capital",
    "recapitalization": "Recapitalization",
}


def _standardise_purpose(purpose) -> str:
    if pd.isna(purpose):
        return "Corp. purposes"
    return _PURPOSE_MAP.get(str(purpose).lower().strip(), "Corp. purposes")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_dealscan() -> pd.DataFrame:
    path = RAW / "dealscan_facilities.csv"
    df = pd.read_csv(
        path,
        parse_dates=["origination_date", "maturity"],
        low_memory=False,
    )
    df = df[df["aisd_bps"].between(0, 2000)].copy()
    df["facilityid"] = df["facilityid"].astype(str).str.strip()
    df["loan_purpose"] = df["loan_purpose"].apply(_standardise_purpose)
    log.info("DealScan: %d facilities loaded", len(df))
    return df


def load_chava_roberts() -> pd.DataFrame:
    path = RAW / "chava_roberts_link.csv"
    df = pd.read_csv(path, dtype=str)
    df["facilityid"] = df["facilityid"].str.strip()
    df["gvkey"] = df["gvkey"].str.strip().str.zfill(6)
    df = df[["facilityid", "gvkey"]].drop_duplicates(subset=["facilityid"])
    log.info("Chava-Roberts link: %d facility-firm pairs", len(df))
    return df


def load_compustat() -> pd.DataFrame:
    path = RAW / "compustat_funda.csv"
    df = pd.read_csv(
        path, parse_dates=["fiscal_year_end"], low_memory=False
    )
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    df = df.rename(columns=COMPUSTAT_RENAME)

    if "ebitda" not in df.columns or df["ebitda"].isna().all():
        df["ebitda"] = np.nan

    df = (
        df.sort_values(["gvkey", "fiscal_year_end"])
        .drop_duplicates(subset=["gvkey", "fiscal_year_end"], keep="last")
    )
    log.info("Compustat: %d firm-year observations", len(df))
    return df


def load_crsp() -> pd.DataFrame:
    path = RAW / "crsp_monthly.csv"
    df = pd.read_csv(path, parse_dates=["crsp_date"], low_memory=False)
    df["permno"] = (
        pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    )
    df["monthly_return"] = pd.to_numeric(
        df["monthly_return"], errors="coerce"
    )
    df = df.sort_values(["permno", "crsp_date"])
    log.info("CRSP monthly: %d observations", len(df))
    return df


def load_ccm_link() -> pd.DataFrame:
    path = RAW / "crsp_ccm_link.csv"
    df = pd.read_csv(path, low_memory=False)
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    df["permno"] = (
        pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    )
    df["linkdt"] = pd.to_datetime(df["linkdt"], errors="coerce")
    df["linkenddt"] = pd.to_datetime(
        df["linkenddt"], errors="coerce"
    ).fillna(pd.Timestamp("2099-12-31"))
    log.info("CCM link: %d gvkey-permno pairs", len(df))
    return df


# ---------------------------------------------------------------------------
# Step 2 — attach gvkey via Chava-Roberts link
# ---------------------------------------------------------------------------

def attach_gvkey(loans: pd.DataFrame, link: pd.DataFrame) -> pd.DataFrame:
    before = len(loans)
    loans = loans.merge(link, on="facilityid", how="left")
    matched = loans["gvkey"].notna().sum()
    log.info(
        "Chava-Roberts join: %d of %d loans matched to a Compustat gvkey",
        matched, before,
    )
    return loans


# ---------------------------------------------------------------------------
# Step 3 — as-of fiscal year merge (no lookahead bias)
# ---------------------------------------------------------------------------

def match_fiscal_year(
    loans: pd.DataFrame, compustat: pd.DataFrame
) -> pd.DataFrame:
    """
    For each loan, find the most-recent Compustat fiscal year whose end
    date falls strictly before origination_date.

    Uses pd.merge_asof for efficiency (O(n log n) instead of row-by-row).
    """
    loans_matched = loans.dropna(subset=["gvkey"]).copy()
    loans_matched = loans_matched.sort_values("origination_date")
    compustat = compustat.sort_values("fiscal_year_end")

    merged = pd.merge_asof(
        loans_matched,
        compustat,
        left_on="origination_date",
        right_on="fiscal_year_end",
        by="gvkey",
        direction="backward",
        suffixes=("", "_comp"),
    )

    merged = merged[
        merged["fiscal_year_end"].notna()
        & (merged["fiscal_year_end"] < merged["origination_date"])
    ]
    merged = merged.loc[:, ~merged.columns.duplicated()]

    log.info(
        "As-of FY merge: %d loans with Compustat financials "
        "(of %d with gvkey)",
        len(merged), len(loans_matched),
    )
    return merged


# ---------------------------------------------------------------------------
# Step 4 — attach CRSP equity features
# ---------------------------------------------------------------------------

def _compute_equity_features(crsp: pd.DataFrame) -> pd.DataFrame:
    crsp = crsp.copy()
    crsp["eq_vol_12m"] = (
        crsp.groupby("permno", observed=True)["monthly_return"]
        .transform(
            lambda x: x.rolling(12, min_periods=6).std() * np.sqrt(12)
        )
    )
    crsp["market_cap_bn"] = crsp["market_cap_k"] / 1e6
    crsp["_eq_month"] = (
        crsp["crsp_date"].dt.to_period("M").dt.to_timestamp("M")
    )
    return crsp[
        ["permno", "_eq_month", "market_cap_bn", "eq_vol_12m"]
    ].copy()


def _get_permno_at_origination(
    panel: pd.DataFrame, ccm: pd.DataFrame
) -> pd.DataFrame:
    expanded = panel[["facilityid", "gvkey", "origination_date"]].merge(
        ccm[["gvkey", "permno", "linkdt", "linkenddt"]],
        on="gvkey",
        how="left",
    )
    no_link = expanded["linkdt"].isna()
    in_range = (
        (expanded["linkdt"] <= expanded["origination_date"])
        & (expanded["linkenddt"] >= expanded["origination_date"])
    )
    valid = expanded[no_link | in_range].copy()
    valid = (
        valid.sort_values("linkdt", na_position="first")
        .drop_duplicates(subset="facilityid", keep="last")
        [["facilityid", "permno"]]
    )
    return panel.merge(valid, on="facilityid", how="left")


def attach_equity(
    panel: pd.DataFrame, crsp: pd.DataFrame, ccm: pd.DataFrame
) -> pd.DataFrame:
    eq_feats = _compute_equity_features(crsp)
    panel = _get_permno_at_origination(panel, ccm)

    panel["_eq_month"] = (
        (panel["origination_date"].dt.to_period("M") - 1)
        .dt.to_timestamp("M")
    )
    panel["permno"] = pd.to_numeric(
        panel["permno"], errors="coerce"
    ).astype("Int64")
    eq_feats["permno"] = pd.to_numeric(
        eq_feats["permno"], errors="coerce"
    ).astype("Int64")

    panel = panel.merge(
        eq_feats, on=["permno", "_eq_month"], how="left"
    )
    panel.drop(
        columns=["_eq_month", "permno"], inplace=True, errors="ignore"
    )
    log.info(
        "Equity attached: %d of %d loans have eq_vol_12m",
        panel["eq_vol_12m"].notna().sum(), len(panel),
    )
    return panel


# ---------------------------------------------------------------------------
# Step 5 — attach FRED macro
# ---------------------------------------------------------------------------

def _fetch_fred_series(series_id: str) -> pd.Series:
    """
    Download a FRED series via the public CSV endpoint (no API key needed).
    """
    from io import StringIO
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    # Normalise: first column is always the date, second is the value
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date")
    return df[series_id]


def attach_fred_spreads(panel: pd.DataFrame) -> pd.DataFrame:
    log.info("Fetching FRED macro series (no API key required)...")
    baa = _fetch_fred_series("BAA").rename("baa")
    aaa = _fetch_fred_series("AAA").rename("aaa")
    rf = _fetch_fred_series("GS10").rename("risk_free_10y")

    macro = pd.concat([baa, aaa, rf], axis=1)
    macro["baa_aaa_spread"] = (macro["baa"] - macro["aaa"]) * 100
    macro = macro.resample("MS").mean()
    macro.index.name = "fred_date"
    macro = macro.reset_index()

    panel = panel.copy()
    panel["fred_date"] = (
        panel["origination_date"].dt.to_period("M").dt.to_timestamp()
    )
    panel = panel.merge(
        macro[["fred_date", "baa_aaa_spread", "risk_free_10y"]],
        on="fred_date",
        how="left",
    )
    panel.drop(columns=["fred_date"], inplace=True, errors="ignore")
    log.info(
        "FRED macro attached; missing baa_aaa_spread: %d",
        panel["baa_aaa_spread"].isna().sum(),
    )
    return panel


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_panel() -> pd.DataFrame:
    loans = load_dealscan()
    link = load_chava_roberts()
    compustat = load_compustat()
    crsp = load_crsp()
    ccm = load_ccm_link()

    loans = attach_gvkey(loans, link)
    panel = match_fiscal_year(loans, compustat)
    panel = attach_equity(panel, crsp, ccm)
    panel = attach_fred_spreads(panel)

    panel = panel.loc[:, ~panel.columns.duplicated()]
    panel.to_parquet(PANEL_OUT, index=False)
    log.info(
        "Panel written -> %s  (%d rows, %d cols)",
        PANEL_OUT, len(panel), panel.shape[1],
    )
    return panel


if __name__ == "__main__":
    build_panel()
