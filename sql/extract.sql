-- =============================================================================
-- WRDS PostgreSQL extraction queries (OPTIONAL — requires WRDS subscription)
-- =============================================================================
-- The primary pipeline in this project uses free public APIs:
--   • SEC EDGAR 8-K filings  → loan spread / facility data  (src/edgar_loans.py)
--   • SEC EDGAR XBRL API     → borrower financials           (src/merge.py)
--   • yfinance               → equity market data            (src/merge.py)
--   • FRED API               → macro controls                (src/merge.py)
--
-- If you later gain WRDS access (e.g. via university library), uncomment
-- and run the queries below to replace the EDGAR/yfinance sources with
-- higher-coverage DealScan + Compustat data.  The downstream feature
-- engineering and modelling code is data-source agnostic as long as the
-- output DataFrame preserves the same column names.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. DealScan facilities  →  data/raw/dealscan_facilities.csv
-- ---------------------------------------------------------------------------
/*
\copy (
    SELECT
        f.facilityid,
        f.packageid,
        p.borrowercompanyid,
        bc.companyname                          AS borrower_name,
        bc.sic_code,
        p.deal_date                             AS origination_date,
        f.facilityamt                           AS facility_amount_mm,
        f.maturity,
        f.facilitytype,
        f.primarypurpose                        AS loan_purpose,
        f.allindrawn                            AS aisd_bps,
        CASE WHEN f.secured = 'Yes' THEN 1
             WHEN f.secured = 'No'  THEN 0
             ELSE NULL END                      AS collateral_flag,
        COALESCE(cov.n_covenants, 0)            AS covenant_count,
        CASE WHEN f.facilitytype ILIKE '%revolv%' THEN 1
             ELSE 0 END                         AS is_revolver
    FROM   lpc.facility         f
    JOIN   lpc.package          p  ON p.packageid        = f.packageid
    JOIN   lpc.borrowercompany  bc ON bc.companyid       = p.borrowercompanyid
    LEFT JOIN (
        SELECT   facilityid, COUNT(DISTINCT covenanttype) AS n_covenants
        FROM     lpc.financialcovenant
        GROUP BY facilityid
    ) cov ON cov.facilityid = f.facilityid
    WHERE  f.currency      = 'USD'
      AND  f.allindrawn    IS NOT NULL
      AND  f.facilityamt   > 0
      AND  p.deal_date     IS NOT NULL
      AND  f.allindrawn BETWEEN 0 AND 2000
) TO 'data/raw/dealscan_facilities.csv' WITH CSV HEADER;
*/

-- ---------------------------------------------------------------------------
-- 2. Chava-Roberts linktable  →  data/raw/chava_roberts_link.csv
-- ---------------------------------------------------------------------------
/*
\copy (
    SELECT bcoid, gvkey, score
    FROM   dealscan.bclink
    WHERE  score >= 1
) TO 'data/raw/chava_roberts_link.csv' WITH CSV HEADER;
*/

-- ---------------------------------------------------------------------------
-- 3. Compustat funda  →  data/raw/compustat_funda.csv
-- ---------------------------------------------------------------------------
/*
\copy (
    SELECT gvkey, datadate AS fiscal_year_end, fyear,
           at, dltt, dlc, ebitda, xint, sale, ppent, che,
           ceq, ni, wcap, re, sich
    FROM   comp.funda
    WHERE  indfmt='INDL' AND datafmt='STD' AND popsrc='D' AND consol='C'
      AND  at > 0 AND datadate BETWEEN '1989-01-01' AND '2022-12-31'
) TO 'data/raw/compustat_funda.csv' WITH CSV HEADER;
*/

-- ---------------------------------------------------------------------------
-- 4. CRSP monthly  →  data/raw/crsp_monthly.csv
-- ---------------------------------------------------------------------------
/*
\copy (
    SELECT permno, date AS crsp_date, ABS(prc) AS price,
           shrout AS shares_outstanding_k, ret AS monthly_return,
           ABS(prc)*shrout AS market_cap_k
    FROM   crsp.msf
    WHERE  date BETWEEN '1989-01-01' AND '2022-12-31'
      AND  shrout > 0 AND prc IS NOT NULL
) TO 'data/raw/crsp_monthly.csv' WITH CSV HEADER;
*/
