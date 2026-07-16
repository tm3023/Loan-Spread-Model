"""
convert_chava_roberts.py — Convert the Chava-Roberts DealScan-Compustat link
from Excel format to CSV.

Input : data/raw/ds-cs-link-post-202401.xlsx  (sheet: links)
Output: data/raw/chava_roberts_link.csv
    Key columns:
        facilityid  — DealScan facility ID (join key to DealScan facilities)
        gvkey       — Compustat firm identifier (6-digit, zero-padded)

Usage:
    python src/convert_chava_roberts.py
"""

import logging
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

XLSX_IN = RAW / "ds-cs-link-post-202401.xlsx"
CSV_OUT = RAW / "chava_roberts_link.csv"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    if not XLSX_IN.exists():
        raise FileNotFoundError(f"Link file not found at {XLSX_IN}")

    log.info("Reading sheet 'links' from %s ...", XLSX_IN)
    df = pd.read_excel(XLSX_IN, sheet_name="links", dtype=str)
    log.info("Columns: %s", list(df.columns))
    log.info("Raw rows: %d", len(df))

    df.columns = [c.lower().strip() for c in df.columns]

    # Normalise gvkey to 6-digit zero-padded string
    df["gvkey"] = df["gvkey"].astype(str).str.strip().str.zfill(6)
    df["facilityid"] = df["facilityid"].astype(str).str.strip()

    # Keep manually verified (NaN score) and probabilistic matches >= 0.5
    if "confidence_score" in df.columns:
        before = len(df)
        scores = pd.to_numeric(df["confidence_score"], errors="coerce")
        df = df[scores.isna() | (scores >= 0.5)]
        log.info(
            "Confidence filter: kept %d of %d rows",
            len(df), before,
        )

    df = df.drop_duplicates(subset=["facilityid"])

    candidates = ["facilityid", "gvkey", "borrowercompanyid", "confidence_score"]
    keep = [c for c in candidates if c in df.columns]
    df[keep].to_csv(CSV_OUT, index=False)
    log.info("Saved %d link rows -> %s", len(df), CSV_OUT)


if __name__ == "__main__":
    main()
