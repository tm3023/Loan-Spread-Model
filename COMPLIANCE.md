# Data Compliance & Sharing Policy

This repository contains **code only**. All input data is licensed and excluded from version
control via `.gitignore`. This document records exactly which files may be published publicly
and which must remain local.

---

## Safe to Commit (Public)

| File / Pattern | Why it is safe |
|---|---|
| `src/*.py` | Original code authored by the project owner — no data embedded |
| `case_studies/toys_r_us.ipynb` | Uses only SEC EDGAR (public domain) and FRED (public domain) data; WRDS data does not appear anywhere in the notebook |
| `notebooks/eda.ipynb` | Code and markdown only; cell outputs stripped before commit |
| `sql/extract.sql` | Query logic only — contains SQL field names, not data values |
| `figures/*.png` | Aggregate SHAP visualisations and PDP curves; model outputs, not raw data; no company identifiers, no dollar-denominated values |
| `results/model_metrics.csv` | Four-row table of RMSE/MAE/R²; model performance statistics, not data |
| `README.md` | Contains only methodology, performance metrics, and schema descriptions — no raw records |
| `COMPLIANCE.md` | This file |
| `requirements.txt` | Package list |
| `.gitignore` | Configuration |
| `data/raw/.gitkeep` | Empty directory marker |

---

## Must NOT Commit (Local Only)

| File | Data owner | Restriction |
|---|---|---|
| `data/raw/dealscan_facilities.csv` | LPC / Refinitiv (via WRDS) | WRDS Subscriber Agreement §4 — redistribution of data in any form is prohibited |
| `data/raw/compustat_funda.csv` | S&P Global (via WRDS) | Same |
| `data/raw/crsp_monthly.csv` | CRSP (via WRDS) | Same |
| `data/raw/crsp_ccm_link.csv` | CRSP (via WRDS) | Same |
| `data/raw/ds-cs-link-post-202401.xlsx` | Michael Roberts / Wharton | No explicit redistribution licence; cite and link only |
| `data/raw/chava_roberts_link.csv` | Derived from above | Same restriction applies to derived files |
| `data/raw/panel.parquet` | Derived from WRDS data | Derivative works of licensed data remain restricted |
| `data/raw/features.parquet` | Derived from WRDS data | Same |
| `data/raw/models/*.pkl`, `*.ubj` | Trained on WRDS data | Out of caution — model weights carry implicit information about the training distribution |
| Any `*.csv`, `*.parquet`, `*.xlsx` in `data/` | Various | Covered by `.gitignore` blanket rules |

---

## Why the Published Content is Permitted

**Performance metrics (RMSE, MAE, R²)** — These are computed statistics describing model
accuracy. They do not allow reconstruction of any underlying data record. Academic journals
and industry practitioners publish such metrics routinely without restriction.

**Sample size counts (11,477 observations, 2010-2022)** — Reporting the number of records
used is standard methodology disclosure. WRDS restricts redistribution of data, not
meta-information about the dataset size.

**SHAP figures** — SHAP values are outputs of the trained model applied to test-set
predictions. They show model behaviour, not raw data. No figure contains company names,
identifiers (GVKEY, permno), or dollar-denominated raw fields.

**PDP figures** — Partial dependence plots show the model's average prediction as a function
of one feature. The x-axis represents leverage or log(assets) as a continuous variable,
not a list of individual company values. Published papers using WRDS routinely include
such plots.

**Case study notebook** — The Toys R Us notebook loads financial data exclusively from
SEC EDGAR (`data.sec.gov/api/xbrl/companyfacts/`) — a public-domain API operated by the
U.S. government with no usage restrictions. FRED macro data is similarly public domain.
The XGBoost model is loaded from local disk (not committed); its predictions are model
outputs, not WRDS records.

---

## WRDS Terms of Use Reference

WRDS Subscriber Agreement: [wrds.wharton.upenn.edu/pages/wrds-subscriber-agreement.html](https://wrds.wharton.upenn.edu/pages/wrds-subscriber-agreement.html)

Key section: **§4 Restrictions on Use** — subscribers may not "distribute, disclose, sell,
resell, sublicense, transmit, or otherwise transfer any Data to any third party."

Code, figures derived from analysis, and model performance statistics are not "Data" as
defined in the agreement and may be shared publicly.
