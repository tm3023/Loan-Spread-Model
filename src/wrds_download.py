"""
wrds_download.py — Download all WRDS data needed for the loan spread model.

Downloads four datasets:
    1. DealScan facility data        -> data/raw/dealscan_facilities.csv
    2. Compustat annual fundamentals  -> data/raw/compustat_funda.csv
    3. CRSP monthly stock file        -> data/raw/crsp_monthly.csv
    4. CRSP-Compustat CCM link table  -> data/raw/crsp_ccm_link.csv

Prerequisites:
    pip install wrds
    WRDS subscription covering: DealScan, Compustat, CRSP

Usage:
    python src/wrds_download.py
    (prompts for WRDS username/password on first run; cached thereafter)
"""

import logging
from pathlib import Path

import pandas as pd
import wrds
from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

LOAN_START = "2010-01-01"
LOAN_END = "2022-12-31"
DATA_START = "2005-01-01"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _query(db: wrds.Connection, sql: str) -> pd.DataFrame:
    """Execute SQL and return a DataFrame (SQLAlchemy 2.x safe)."""
    result = db.connection.execute(text(sql))
    return pd.DataFrame(result.fetchall(), columns=result.keys())


def download_dealscan(db: wrds.Connection) -> None:
    """
    Pull one row per facility from dealscan.dealscan (the denormalized view).
    DISTINCT ON ensures one row per lpc_tranche_id even though the view has
    one row per lender.
    """
    log.info(
        "Downloading DealScan facilities (%s - %s)...", LOAN_START, LOAN_END
    )
    sql = f"""
        SELECT DISTINCT ON (d.lpc_tranche_id)
            d.lpc_tranche_id                AS facilityid,
            d.lpc_deal_id                   AS packageid,
            d.borrower_id                   AS borrowercompanyid,
            d.borrower_name,
            d.sic_code,
            d.deal_active_date              AS origination_date,
            d.tranche_amount                AS facility_amount_mm,
            d.tranche_maturity_date         AS maturity,
            d.tranche_type                  AS facilitytype,
            d.primary_purpose               AS loan_purpose,
            d.all_in_spread_drawn_bps       AS aisd_bps,
            CASE WHEN d.secured = 'Yes' THEN 1
                 WHEN d.secured = 'No'  THEN 0
                 ELSE NULL END              AS collateral_flag,
            COALESCE(cov.n_covenants, 0)   AS covenant_count,
            CASE WHEN d.tranche_type ILIKE '%revolv%' THEN 1
                 ELSE 0 END                AS is_revolver
        FROM dealscan.dealscan d
        LEFT JOIN (
            SELECT   packageid,
                     COUNT(DISTINCT covenanttype) AS n_covenants
            FROM     dealscan.financialcovenant
            GROUP BY packageid
        ) cov ON cov.packageid::text = d.lpc_deal_id
        WHERE d.deal_currency              = 'U.S. Dollar'
          AND d.all_in_spread_drawn_bps   IS NOT NULL
          AND d.tranche_amount             > 0
          AND d.deal_active_date          IS NOT NULL
          AND d.all_in_spread_drawn_bps   BETWEEN 0 AND 2000
          AND d.deal_active_date
              BETWEEN '{LOAN_START}' AND '{LOAN_END}'
        ORDER BY d.lpc_tranche_id
    """
    df = _query(db, sql)
    out = RAW / "dealscan_facilities.csv"
    df.to_csv(out, index=False)
    log.info("DealScan: %d rows -> %s", len(df), out)


def download_compustat(db: wrds.Connection) -> None:
    log.info(
        "Downloading Compustat annual fundamentals (%s - %s)...",
        DATA_START, LOAN_END,
    )
    sql = f"""
        SELECT gvkey,
               datadate  AS fiscal_year_end,
               fyear,
               at,
               dltt,
               dlc,
               ebitda,
               xint,
               sale,
               ppent,
               che,
               ceq,
               ni,
               wcap,
               re,
               sich
        FROM   comp.funda
        WHERE  indfmt  = 'INDL'
          AND  datafmt = 'STD'
          AND  popsrc  = 'D'
          AND  consol  = 'C'
          AND  at      > 0
          AND  datadate BETWEEN '{DATA_START}' AND '{LOAN_END}'
    """
    df = _query(db, sql)
    out = RAW / "compustat_funda.csv"
    df.to_csv(out, index=False)
    log.info("Compustat: %d rows -> %s", len(df), out)


def download_crsp(db: wrds.Connection) -> None:
    log.info(
        "Downloading CRSP monthly stock file (%s - %s)...",
        DATA_START, LOAN_END,
    )
    sql = f"""
        SELECT permno,
               date             AS crsp_date,
               ABS(prc)        AS price,
               shrout           AS shares_outstanding_k,
               ret              AS monthly_return,
               ABS(prc)*shrout  AS market_cap_k
        FROM   crsp.msf
        WHERE  date    BETWEEN '{DATA_START}' AND '{LOAN_END}'
          AND  shrout  > 0
          AND  prc     IS NOT NULL
    """
    df = _query(db, sql)
    out = RAW / "crsp_monthly.csv"
    df.to_csv(out, index=False)
    log.info("CRSP monthly: %d rows -> %s", len(df), out)


def download_ccm_link(db: wrds.Connection) -> None:
    log.info("Downloading CRSP-Compustat CCM link table...")
    sql = """
        SELECT gvkey,
               lpermno   AS permno,
               linkdt,
               linkenddt,
               linktype,
               linkprim
        FROM   crsp.ccmxpf_linktable
        WHERE  linktype IN ('LU', 'LC')
          AND  linkprim IN ('P', 'C')
    """
    df = _query(db, sql)
    out = RAW / "crsp_ccm_link.csv"
    df.to_csv(out, index=False)
    log.info("CCM link: %d rows -> %s", len(df), out)


def main() -> None:
    log.info("Connecting to WRDS (enter credentials when prompted)...")
    db = wrds.Connection()
    try:
        download_dealscan(db)
        download_compustat(db)
        download_crsp(db)
        download_ccm_link(db)
        log.info("All WRDS downloads complete. Files saved to %s", RAW)
    finally:
        db.close()


if __name__ == "__main__":
    main()
