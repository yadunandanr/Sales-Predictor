"""Unit tests for Person 1 data cleaning & chronological split pipeline."""

from __future__ import annotations

import pandas as pd
import pytest

from preprocessing.data_cleaning import (
    DATE_COL,
    PRODUCT_COL,
    SALES_COL,
    STORE_COL,
    chronological_split,
    handle_missing_values,
    handle_outliers,
    parse_and_filter_dates,
    remove_duplicates,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            DATE_COL: [
                "2016-05-01",
                "2023-12-31",
                "2024-01-01",
                "2025-06-15",
                "invalid-date",
                "2014-01-01",
                "2016-05-01",
            ],
            STORE_COL: ["S1", "S1", "S2", "S2", "S1", "S1", "S1"],
            PRODUCT_COL: ["P1", "P2", "P1", "P2", "P1", "P1", "P1"],
            SALES_COL: [100.0, 200.0, 300.0, 400.0, 50.0, 80.0, 100.0],
            "discount_pct": [0.0, None, 10.0, 5.0, 0.0, 0.0, 0.0],
        }
    )


def test_parse_and_filter_dates_drops_invalid_and_out_of_range():
    df = _sample_df()
    parsed, invalid_cnt, out_of_range_cnt = parse_and_filter_dates(df)
    assert invalid_cnt == 1
    assert out_of_range_cnt == 1
    assert pd.api.types.is_datetime64_any_dtype(parsed[DATE_COL])
    assert parsed[DATE_COL].min() >= pd.Timestamp("2016-01-01")
    assert parsed[DATE_COL].max() <= pd.Timestamp("2025-12-31")


def test_remove_duplicates():
    df = _sample_df()
    parsed, _, _ = parse_and_filter_dates(df)
    deduped, dup_cnt = remove_duplicates(parsed)
    assert dup_cnt == 1
    assert deduped.duplicated().sum() == 0


def test_handle_missing_values_drops_critical_and_imputes_non_critical():
    df = _sample_df()
    parsed, _, _ = parse_and_filter_dates(df)
    deduped, _ = remove_duplicates(parsed)
    deduped.loc[deduped.index[0], SALES_COL] = None

    cleaned, crit_dropped, imputed = handle_missing_values(deduped)
    assert crit_dropped == 1
    assert imputed == 1
    assert cleaned.isna().sum().sum() == 0


def test_handle_outliers_removes_negative_and_caps_extreme():
    df = pd.DataFrame(
        {
            DATE_COL: pd.date_range("2020-01-01", periods=10, freq="D"),
            SALES_COL: [-10.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 99999.0],
        }
    )
    cleaned, neg_cnt, capped_cnt, upper_cap = handle_outliers(df, upper_quantile=0.9)
    assert neg_cnt == 1
    assert capped_cnt >= 1
    assert (cleaned[SALES_COL] >= 0).all()
    assert cleaned[SALES_COL].max() <= upper_cap


def test_chronological_split_has_zero_leakage():
    df = pd.DataFrame(
        {
            DATE_COL: pd.to_datetime(["2016-01-01", "2023-12-31", "2024-01-01", "2025-12-31"]),
            SALES_COL: [10.0, 20.0, 30.0, 40.0],
        }
    )
    train_df, test_df = chronological_split(df)
    assert len(train_df) == 2
    assert len(test_df) == 2
    assert train_df[DATE_COL].max() < test_df[DATE_COL].min()
