# Syndicated Loan Spread Prediction

> Predicting the all-in drawn spread (AISD) on syndicated loans using institutional-grade
> financial data, gradient-boosted trees, and rigorous no-lookahead feature engineering.
> XGBoost achieves **R² = 0.45** on a held-out 2020–2022 test set.

---

## ⚠ Data & Licensing Notice

**This repository contains code only.** All input datasets are licensed and excluded from
version control by `.gitignore`.

| Dataset | Licence holder | How to obtain |
|---|---|---|
| LPC DealScan | Refinitiv / LPC via WRDS | University/institutional WRDS subscription |
| Compustat Fundamentals Annual | S&P Global via WRDS | University/institutional WRDS subscription |
| CRSP Monthly Stock File | CRSP via WRDS | University/institutional WRDS subscription |
| Chava-Roberts DealScan-Compustat Link | Michael Roberts (Wharton) | [Roberts' data page](https://finance.wharton.upenn.edu/~mrrobert/) |
| FRED macro series (BAA, AAA, GS10) | Federal Reserve | Public domain; fetched at runtime |

See [COMPLIANCE.md](COMPLIANCE.md) for a detailed breakdown of what is and is not permitted
to be shared publicly under the WRDS Subscriber Agreement.

---

## Project Overview

The all-in drawn spread (AISD) on a syndicated loan is the market's aggregate price of
borrower credit risk. It reflects default probability, information asymmetry between lender
and borrower, and prevailing conditions in the credit market. Modelling AISD is therefore
relevant across several areas of practice:

- **Credit portfolio management**: pricing new facilities and marking existing positions to market
- **CLO structuring**: estimating the cost of debt across heterogeneous collateral pools
- **Distress early-warning**: identifying balance-sheet deterioration before covenant breaches

This project draws on spread determinants documented in Ivashina (2009) and Berg, Saunders &
Steffen (2016), using institutional WRDS data: LPC DealScan for loan terms, Compustat for
annual borrower financials, and CRSP for equity market signals. The central finding is that
gradient-boosted trees capture spread-relevant non-linearities, particularly interactions
between leverage, firm size, and macro credit conditions, that OLS benchmarks cannot represent.

---

## Engineering Highlights

The following documents the non-obvious technical decisions in this implementation.

### 1. Server-side SQL with SQLAlchemy 2.x compatibility

DealScan data is extracted via server-side SQL against the WRDS PostgreSQL cluster. All
filtering by currency, spread range, and date window is applied at the database layer,
before any data is transferred over the network. The `DISTINCT ON (lpc_tranche_id)` clause
deduplicates the denormalised lender-share view to one row per facility, also on the
database side.

WRDS 3.5 ships with SQLAlchemy 2.x, which removed the legacy `engine.execute()` interface.
Compatibility is achieved as follows:

```python
result = db.connection.execute(text(sql))
df = pd.DataFrame(result.fetchall(), columns=result.keys())
```

A cross-type join between the covenant subquery (numeric `packageid`) and the DealScan view
(varchar `lpc_deal_id`) requires an explicit `::text` cast; without it, PostgreSQL raises
`operator does not exist: numeric = character varying`.

### 2. Facility-level entity resolution via Chava-Roberts

Borrower identities are resolved using the DealScan-Compustat Linking Database maintained by
Michael Roberts (the data product associated with Chava & Roberts (2008)), which maps DealScan
`facilityid` to Compustat `gvkey`. Matching is performed at the facility level rather than
collapsing to one row per borrower: a single borrower's facilities all resolve to the same
`gvkey`, but each retains its own loan-specific terms (pricing, maturity, covenants) as a
separate observation.

Two implementation details are worth noting. Links with a confidence score below 0.5 are
dropped; however, `NaN` scores denote manually verified links and are retained, since
filtering on `>= 0.5` alone would incorrectly discard the highest-quality matches. Additionally,
`gvkey` is zero-padded to six characters via `str.zfill(6)` before any cross-source join,
to prevent silent mismatches arising from inconsistent leading-zero conventions across WRDS fields.

### 3. No-lookahead-bias enforcement

Every feature must be observable at the time the loan is priced. The as-of fiscal year merge
uses `pd.merge_asof(direction='backward')` to match each loan to the most-recent Compustat
annual report whose `datadate` falls strictly before `origination_date`:

```python
merged = pd.merge_asof(
    loans.sort_values("origination_date"),
    compustat.sort_values("fiscal_year_end"),
    left_on="origination_date",
    right_on="fiscal_year_end",
    by="gvkey",
    direction="backward",
)
```

A secondary filter removes any rows where `fiscal_year_end >= origination_date`, handling
edge cases at period boundaries. All imputation medians and z-score scaler statistics are
computed on the training split only; they are then applied, without re-estimation, to the
validation and test sets.

### 4. Winsorisation at training-set quantiles

All ratio features are winsorised at the 1st and 99th percentile of the training
distribution, not the full sample. The clip bounds are estimated on training-period loans
only (`origination_date <= TRAIN_END`) and then applied to every split:

```python
train = df["origination_date"] <= TRAIN_END
lo, hi = df.loc[train, col].quantile([0.01, 0.99])
df[col] = df[col].clip(lo, hi)
```

This removes the influence of extreme outliers without discarding observations. Crucially,
the validation and test loans play no part in setting the thresholds, preserving the
integrity of the out-of-sample evaluation.

### 5. Time-aware chronological split

Observations are partitioned by `origination_date` rather than randomly shuffled. Random
shuffling would allow future macro conditions to leak into the training set, inflating
apparent performance. The split boundaries are chosen to isolate a genuine stress period
as the test set:

| Split | Period | Observations |
|---|---|---|
| Train | 2010–2016 | 7,841 |
| Validate | 2017–2019 | 2,268 |
| Test | 2020–2022 | 1,368 |

XGBoost early stopping is monitored on the validation set with a patience of 50 rounds.
The validation set is not used for hyperparameter search; it therefore remains an
independent check on out-of-sample generalisation.

### 6. FRED macro via public CSV endpoint

Monthly BAA, AAA, and 10-year Treasury yields are fetched at runtime from FRED's public
CSV endpoint. No API key is required:

```python
url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
```

Column names are normalised explicitly on ingestion, since FRED CSV headers vary by series.
Numeric coercion with `errors='coerce'` handles the `"."` sentinel that FRED uses to
denote missing observations.

### 7. SHAP explainability

`shap.TreeExplainer` is applied to the fitted XGBoost model on the held-out test set. SHAP
values additively decompose each prediction into individual feature contributions, without the
sampling approximation that model-agnostic SHAP estimators require.

By mean |SHAP value|, loan structural terms dominate: `log_facility_amount`, `collateral_flag`,
and `is_revolver` rank above every borrower balance-sheet ratio, including `leverage` and
`altman_z`. This is consistent with how syndicated loans are actually priced — facility size and
security status map fairly directly onto the spread grid lenders quote — while credit-fundamental
features contribute smaller, second-order effects on top of that structural baseline. The partial
dependence plots below show those second-order effects are still directionally sensible: the SHAP
contribution from leverage rises sharply above roughly 0.6–0.7, and the largest borrowers (highest
`log_firm_size`) see a distinctly negative spread contribution — though firm size is not simply
monotonic across the full range; the effect is concentrated at the top end.

![SHAP Summary](figures/shap_summary_beeswarm.png)

---

## Model Architecture & Results

### Models

| Model | Description |
|---|---|
| OLS | statsmodels; HC3 heteroskedasticity-robust standard errors |
| Lasso | scikit-learn; alpha selected by validation-set RMSE |
| Ridge | scikit-learn; alpha selected by validation-set RMSE |
| XGBoost | Gradient-boosted trees; early stopping on validation set |

Missing values are median-imputed and all features are z-score scaled using training-set
statistics only. Loan purpose dummies are one-hot encoded without dropping a base category,
as XGBoost handles collinear inputs natively.

### Out-of-sample results (2020–2022 test set)

| Model | RMSE (bps) | MAE (bps) | R² |
|---|---|---|---|
| OLS | 158.44 | 114.03 | 0.370 |
| Lasso | 158.29 | 113.58 | 0.371 |
| Ridge | 158.49 | 114.17 | 0.369 |
| **XGBoost** | **148.04** | **103.71** | **0.450** |

XGBoost achieves approximately a 7% reduction in RMSE relative to OLS, with R² = 0.45
across a test period that encompasses COVID-era credit stress. The performance gap between
XGBoost and the linear benchmarks reflects the contribution of
non-linear interactions, particularly those between leverage and macro credit conditions,
which linear models cannot capture.

Machine-readable metrics: [`results/model_metrics.csv`](results/model_metrics.csv)

---

## Feature Set

| Category | Features |
|---|---|
| **Borrower: balance sheet** | `leverage`, `profitability`, `interest_coverage`, `log_firm_size`, `tangibility`, `cash_ratio` |
| **Borrower: distress signal** | `altman_z` (Altman Z′, private-firm variant), `altman_x1` (liquidity), `altman_x2` (retained earnings), `altman_x3` (EBITDA/assets, a proxy for Altman's EBIT/assets because the Compustat extract has no EBIT field; it overstates X3 by D&A/assets), `altman_x4` (book equity / total liabilities, approximated as assets − book equity), `altman_x5` (asset turnover) |
| **Loan structural** | `log_facility_amount`, `maturity_months`, `collateral_flag`, `covenant_count`, `is_revolver` |
| **Loan purpose** | One-hot: Corp. purposes, Debt repay, Acquisition line, LBO, Working capital, Recapitalization, Other |
| **Market** | `eq_vol_12m`: rolling 12-month annualised equity return volatility (from CRSP) |
| **Macro** | `baa_aaa_spread`: BAA minus AAA yield in basis points; `risk_free_10y`: 10-year Treasury yield (%) |

All ratio features are winsorised at the 1st and 99th percentile of the training distribution.

---

## Data Sources

| Source | Fields used | Access |
|---|---|---|
| **WRDS DealScan** (`dealscan.dealscan`) | `all_in_spread_drawn_bps`, `tranche_amount`, `tranche_maturity_date`, `tranche_type`, `primary_purpose`, `secured`, covenant count | WRDS subscription |
| **WRDS Compustat** (`comp.funda`) | `at`, `dltt`, `dlc`, `ebitda`, `xint`, `sale`, `ppent`, `che`, `ceq`, `ni`, `wcap`, `re` | WRDS subscription |
| **WRDS CRSP** (`crsp.msf`, `crsp.ccmxpf_linktable`) | `ret`, `prc`, `shrout` (monthly); `gvkey` → `permno` link | WRDS subscription |
| **Chava-Roberts link** | `facilityid` → `gvkey` (facility-level, confidence-scored) | [Roberts' Wharton page](https://finance.wharton.upenn.edu/~mrrobert/) |
| **FRED** (`BAA`, `AAA`, `GS10`) | Monthly corporate bond yields and 10-year Treasury yield | Public; fetched at runtime |

---

## Project Structure

```
loan-spread-model/
├── src/
│   ├── wrds_download.py         Download DealScan / Compustat / CRSP from WRDS
│   ├── convert_chava_roberts.py Convert Chava-Roberts Excel link → CSV
│   ├── wrds_merge.py            Panel build: loans + financials + CRSP + FRED
│   ├── features.py              Feature engineering and winsorisation
│   ├── model.py                 OLS / Lasso / Ridge / XGBoost training & evaluation
│   └── shap_analysis.py         SHAP explainability plots
├── case_studies/
│   └── toys_r_us.ipynb          Out-of-sample distress case study (SEC EDGAR data only)
├── notebooks/
│   └── eda.ipynb                Exploratory data analysis
├── sql/
│   └── extract.sql              Reference WRDS SQL queries
├── figures/                     SHAP plots (model outputs; no raw data)
│   ├── shap_summary_beeswarm.png
│   ├── shap_summary_bar.png
│   ├── shap_pdp_leverage.png
│   └── shap_pdp_log_firm_size.png
├── results/
│   └── model_metrics.csv        Out-of-sample RMSE / MAE / R²
├── data/raw/                    Gitignored; all raw and derived data files
│   └── .gitkeep                 Directory marker only
├── COMPLIANCE.md                Data licensing and sharing policy
├── requirements.txt
└── README.md
```

---

## Running the Pipeline

### Prerequisites

```bash
pip install -r requirements.txt
```

A WRDS subscription is required, covering DealScan, Compustat, and CRSP. Access is
typically available through university libraries; contact your institution for details.

### Step 1: Chava-Roberts link table

1. Download from [finance.wharton.upenn.edu/~mrrobert/](https://finance.wharton.upenn.edu/~mrrobert/)
   → Research → Data → DealScan-Compustat Linking Database
2. Save the file to `data/raw/ds-cs-link-post-202401.xlsx`
3. Run the conversion script:

```bash
python src/convert_chava_roberts.py
```

### Step 2: Download WRDS data

```bash
python src/wrds_download.py
# Prompts for WRDS username and password
# Downloads four files; the CRSP monthly file is the largest (~3M rows)
```

### Step 3: Build panel, engineer features, train models

```bash
python src/wrds_merge.py       # merges all sources → data/raw/panel.parquet
python src/features.py         # engineers features → data/raw/features.parquet
python src/model.py            # trains and evaluates all models → data/raw/model_comparison.csv
python src/shap_analysis.py    # generates SHAP figures → figures/
```

`model.py` intentionally writes only to the gitignored `data/raw/` directory, never directly to a
tracked path. `results/model_metrics.csv` is a manually reviewed copy of `data/raw/model_comparison.csv`,
committed only after a human confirms it contains aggregate statistics and no data that WRDS's
Subscriber Agreement restricts (see [COMPLIANCE.md](COMPLIANCE.md)). This is deliberate, not an
oversight — no script in this repo writes directly into a version-controlled path.

### Step 4: Explore

```bash
jupyter lab
# Open notebooks/eda.ipynb or case_studies/toys_r_us.ipynb
```

---

## Case Study: Toys "R" Us

[`case_studies/toys_r_us.ipynb`](case_studies/toys_r_us.ipynb) applies the trained XGBoost
model to Toys R Us annual 10-K filings sourced from SEC EDGAR. The company was
taken private in the 2005 KKR LBO and consequently does not appear in the training set;
the case study therefore constitutes a genuine test of out-of-sample generalisation.

EDGAR's structured (XBRL) financial data for the company covers fiscal years ending
January 2009 to January 2016; the first post-LBO years (2005–2008) are not available from
this public source. The notebook prices a hypothetical $500M, 5-year secured term loan for
each year. Over that window the predicted AISD stays in a narrow, elevated band of roughly
360–385 bps, and the two highest predictions (384 and 381 bps) come from the last two years
of filings (fiscal years ending January 2015 and 2016) before the September 2017 bankruptcy
filing. Book leverage rises from 0.58 to 0.68 between the fiscal years ending January 2012
and January 2016. The Altman Z′-score (the book-equity, private-firm variant — Toys R Us
had no public equity after the 2005 LBO, so the classic market-equity Z is not computable
for this company) lies between 1.48 and 1.90 wherever EDGAR reports all five inputs, i.e.
in Altman's grey zone (1.23–2.90) rather than below the 1.23 distress threshold. EDGAR
reports no separate D&A for the company, so X3 here equals EBIT/assets, matching Altman's
definition.

> The case study uses only public SEC EDGAR data (CIK 0001005414) and FRED.
> No WRDS data is accessed or embedded in the notebook.

---

## References

- Ivashina, V. (2009). Asymmetric information effects on loan spreads. *Journal of Financial Economics*, 92(2), 300–319.
- Berg, T., Saunders, A., & Steffen, S. (2016). The total cost of corporate borrowing in the loan market: Don't ignore the fees. *Journal of Finance*, 71(3), 1357–1392.
- Chava, S., & Roberts, M. R. (2008). How does financing impact investment? The role of debt covenants. *Journal of Finance*, 63(5), 2085–2121.
- Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *Journal of Finance*, 23(4), 589–609.
- Altman, E. I. (1983). *Corporate Financial Distress: A Complete Guide to Predicting, Avoiding, and Dealing with Bankruptcy.* John Wiley & Sons. (Source of the Z′ private-firm variant used in `altman_z`, which substitutes book equity for market equity in X4.)
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS*, 30, 4765–4774.
- Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., & Lee, S.-I. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*, 2, 56–67.
