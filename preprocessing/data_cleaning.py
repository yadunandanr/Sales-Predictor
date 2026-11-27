"""Data cleaning and chronological train/test splitting for Sales Predictor (Person 1).

Usage:
    python preprocessing/data_cleaning.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


# ==============================================================================
# CONFIGURATION - Update these column names if switching to a different CSV
# ==============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "sales.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SUMMARY_PATH = PROCESSED_DIR / "cleaning_summary.json"

DATE_COL = "date"
STORE_COL = "store_id"
PRODUCT_COL = "product_id"
SALES_COL = "sales"

PERIOD_START = "2016-01-01"
PERIOD_END = "2025-12-31"
SPLIT_DATE = "2024-01-01"  # 8 years train (2016-2023), 2 years test (2024-2025)


@dataclass
class CleaningReport:
    raw_rows: int
    raw_cols: int
    invalid_dates_dropped: int
    out_of_range_dates_dropped: int
    duplicates_removed: int
    critical_missing_dropped: int
    non_critical_missing_imputed: int
    negative_sales_removed: int
    outliers_capped: int
    outlier_upper_threshold: float
    cleaned_rows: int
    cleaned_cols: int
    train_rows: int
    test_rows: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw sales dataset from CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {path}. "
            "Place your CSV there or run scripts/generate_sample_data.py first."
        )
    return pd.read_csv(path)


def parse_and_filter_dates(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    start_date: str = PERIOD_START,
    end_date: str = PERIOD_END,
) -> tuple[pd.DataFrame, int, int]:
    """Convert date column to datetime64 and keep records within [start_date, end_date]."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    invalid_count = int(out[date_col].isna().sum())
    if invalid_count > 0:
        out = out.dropna(subset=[date_col])

    before_range = len(out)
    mask = (out[date_col] >= pd.Timestamp(start_date)) & (
        out[date_col] <= pd.Timestamp(end_date)
    )
    out = out[mask].copy()
    out_of_range_count = before_range - len(out)
    return out, invalid_count, out_of_range_count


def remove_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove exact duplicate rows."""
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        df = df.drop_duplicates().copy()
    return df, dup_count


def handle_missing_values(
    df: pd.DataFrame,
    critical_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int, int]:
    """Drop rows missing critical columns and impute non-critical nulls safely."""
    if critical_cols is None:
        critical_cols = [DATE_COL, STORE_COL, PRODUCT_COL, SALES_COL]

    out = df.copy()
    before = len(out)
    out = out.dropna(subset=critical_cols).copy()
    critical_dropped = before - len(out)

    imputed_cells = 0
    for col in out.columns:
        null_count = int(out[col].isna().sum())
        if null_count == 0:
            continue
        imputed_cells += null_count
        if pd.api.types.is_numeric_dtype(out[col]):
            # Use training-period median if available to avoid future leakage
            fill_val = None
            if DATE_COL in out.columns:
                train_mask = out[DATE_COL] < pd.Timestamp(SPLIT_DATE)
                train_valid = out.loc[train_mask, col].dropna()
                if not train_valid.empty:
                    fill_val = train_valid.median()
            if fill_val is None or pd.isna(fill_val):
                col_valid = out[col].dropna()
                fill_val = col_valid.median() if not col_valid.empty else 0.0
            out[col] = out[col].fillna(fill_val)
        else:
            out[col] = out[col].fillna("Unknown")

    return out, critical_dropped, imputed_cells


def handle_outliers(
    df: pd.DataFrame,
    sales_col: str = SALES_COL,
    date_col: str = DATE_COL,
    split_date: str = SPLIT_DATE,
    upper_quantile: float = 0.995,
) -> tuple[pd.DataFrame, int, int, float]:
    """Remove negative sales and cap extreme high outliers using training-period quantile."""
    out = df.copy()
    neg_count = int((out[sales_col] < 0).sum())
    if neg_count > 0:
        out = out[out[sales_col] >= 0].copy()

    # Compute threshold on training window only so test distribution never leaks into preprocessing
    if date_col in out.columns:
        train_mask = out[date_col] < pd.Timestamp(split_date)
        ref_series = out.loc[train_mask, sales_col] if train_mask.any() else out[sales_col]
    else:
        ref_series = out[sales_col]

    upper_cap = float(ref_series.quantile(upper_quantile))
    outliers_capped = int((out[sales_col] > upper_cap).sum())
    out[sales_col] = out[sales_col].clip(upper=upper_cap)
    return out, neg_count, outliers_capped, upper_cap


def chronological_split(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    split_date: str = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically into 8 years train (2016-2023) and 2 years test (2024-2025)."""
    ordered = df.sort_values(date_col).reset_index(drop=True)
    cutoff = pd.Timestamp(split_date)
    train_df = ordered[ordered[date_col] < cutoff].copy()
    test_df = ordered[ordered[date_col] >= cutoff].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test split is empty. Verify dataset date coverage.")
    if train_df[date_col].max() >= test_df[date_col].min():
        raise ValueError("Temporal leakage detected between train and test splits.")

    return train_df, test_df


def run_pipeline(
    raw_path: Path = RAW_DATA_PATH,
    processed_dir: Path = PROCESSED_DIR,
) -> CleaningReport:
    """Execute full Person 1 cleaning and chronological split pipeline."""
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(raw_path)
    raw_rows, raw_cols = raw_df.shape

    df, invalid_dates, out_of_range = parse_and_filter_dates(raw_df)
    df, dupes_removed = remove_duplicates(df)
    df, crit_dropped, non_crit_imputed = handle_missing_values(df)
    df, neg_removed, outliers_capped, upper_cap = handle_outliers(df)

    df = df.sort_values([DATE_COL, STORE_COL, PRODUCT_COL]).reset_index(drop=True)
    train_df, test_df = chronological_split(df)

    df.to_csv(processed_dir / "cleaned_sales.csv", index=False)
    train_df.to_csv(processed_dir / "train_sales.csv", index=False)
    test_df.to_csv(processed_dir / "test_sales.csv", index=False)

    report = CleaningReport(
        raw_rows=int(raw_rows),
        raw_cols=int(raw_cols),
        invalid_dates_dropped=invalid_dates,
        out_of_range_dates_dropped=out_of_range,
        duplicates_removed=dupes_removed,
        critical_missing_dropped=crit_dropped,
        non_critical_missing_imputed=non_crit_imputed,
        negative_sales_removed=neg_removed,
        outliers_capped=outliers_capped,
        outlier_upper_threshold=round(upper_cap, 2),
        cleaned_rows=int(len(df)),
        cleaned_cols=int(df.shape[1]),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        train_start=str(train_df[DATE_COL].min().date()),
        train_end=str(train_df[DATE_COL].max().date()),
        test_start=str(test_df[DATE_COL].min().date()),
        test_end=str(test_df[DATE_COL].max().date()),
    )

    with open(processed_dir / "cleaning_summary.json", "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print("=== Person 1: Data Cleaning & Split Summary ===")
    for k, v in asdict(report).items():
        print(f"{k}: {v}")

    return report


if __name__ == "__main__":
    run_pipeline()
