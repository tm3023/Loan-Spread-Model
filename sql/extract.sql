-- =============================================================================
-- WRDS PostgreSQL extraction queries (reference copy)
-- =============================================================================
-- These are the exact queries executed by src/wrds_download.py. They are kept
-- here as a standalone reference for anyone auditing the pipeline without
-- running Python, and must be kept in sync with wrds_download.py if the
-- extraction logic changes.
--
-- Requires a WRDS subscription covering DealScan, Compustat, and CRSP.
-- Query logic only — no data values are stored in this repository. See
-- COMPLIANCE.md for the full data-sharing policy.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. DealScan facilities  →  data/raw/dealscan_facilities.csv
-- ---------------------------------------------------------------------------
/*
\copy (
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
          BETWEEN '2010-01-01' AND '2022-12-31'
    ORDER BY d.lpc_tranche_id
) TO 'data/raw/dealscan_facilities.csv' WITH CSV HEADER;
*/

-- ---------------------------------------------------------------------------
-- 2. Chava-Roberts linktable
-- ---------------------------------------------------------------------------
-- Not queried via SQL: the Roberts DealScan-Compustat Linking Database is
-- distributed as an Excel workbook (see README "Step 1"), not a WRDS table.
-- src/convert_chava_roberts.py converts it to
-- data/raw/chava_roberts_link.csv.

-- ---------------------------------------------------------------------------
-- 3. Compustat funda  →  data/raw/compustat_funda.csv
-- ---------------------------------------------------------------------------
/*
\copy (
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
      AND  datadate BETWEEN '2005-01-01' AND '2022-12-31'
) TO 'data/raw/compustat_funda.csv' WITH CSV HEADER;
*/

-- ---------------------------------------------------------------------------
-- 4. CRSP monthly  →  data/raw/crsp_monthly.csv
-- ---------------------------------------------------------------------------
/*
\copy (
    SELECT permno,
           date             AS crsp_date,
           ABS(prc)        AS price,
           shrout           AS shares_outstanding_k,
           ret              AS monthly_return,
           ABS(prc)*shrout  AS market_cap_k
    FROM   crsp.msf
    WHERE  date    BETWEEN '2005-01-01' AND '2022-12-31'
      AND  shrout  > 0
      AND  prc     IS NOT NULL
) TO 'data/raw/crsp_monthly.csv' WITH CSV HEADER;
*/

-- ---------------------------------------------------------------------------
-- 5. CRSP-Compustat CCM link  →  data/raw/crsp_ccm_link.csv
-- ---------------------------------------------------------------------------
/*
\copy (
    SELECT gvkey,
           lpermno   AS permno,
           linkdt,
           linkenddt,
           linktype,
           linkprim
    FROM   crsp.ccmxpf_linktable
    WHERE  linktype IN ('LU', 'LC')
      AND  linkprim IN ('P', 'C')
) TO 'data/raw/crsp_ccm_link.csv' WITH CSV HEADER;
*/
