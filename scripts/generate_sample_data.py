"""Generate a realistic 120,000-row (1.2 lakh) retail sales dataset (2016-2025)
for local pipeline verification and demonstration.

Note: Replace data/raw/sales.csv with your downloaded Kaggle CSV if your team
selects an external dataset (and update column names in preprocessing/data_cleaning.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "sales.csv"


def generate_dataset(num_rows: int = 120_000, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    stores = [f"STORE_{i:02d}" for i in range(1, 16)]
    regions = {s: ["North", "South", "East", "West", "Central"][i % 5] for i, s in enumerate(stores)}
    categories = ["Electronics", "Groceries", "Apparel", "Home & Kitchen", "Personal Care"]
    cat_base_price = {
        "Electronics": (80.0, 450.0),
        "Groceries": (5.0, 45.0),
        "Apparel": (20.0, 140.0),
        "Home & Kitchen": (25.0, 220.0),
        "Personal Care": (8.0, 65.0),
    }

    products = [f"PROD_{i:03d}" for i in range(1, 51)]
    prod_cat = {p: categories[i % len(categories)] for i, p in enumerate(products)}
    prod_price = {
        p: round(float(rng.uniform(*cat_base_price[prod_cat[p]])), 2) for p in products
    }

    # Date span: 2016-01-01 to 2025-12-31
    start = pd.Timestamp("2016-01-01")
    total_days = (pd.Timestamp("2025-12-31") - start).days + 1
    day_offsets = rng.integers(0, total_days, size=num_rows)
    dates = start + pd.to_timedelta(day_offsets, unit="D")

    chosen_stores = rng.choice(stores, size=num_rows)
    chosen_products = rng.choice(products, size=num_rows)
    chosen_categories = np.array([prod_cat[p] for p in chosen_products])
    chosen_regions = np.array([regions[s] for s in chosen_stores])
    unit_prices = np.array([prod_price[p] for p in chosen_products])

    # Seasonal + growth + store + promo effects
    years_since_2016 = (dates.year - 2016).to_numpy()
    months = dates.month.to_numpy()
    dow = dates.day_of_week.to_numpy()

    seasonal_boost = np.where(np.isin(months, [11, 12]), 1.45, np.where(np.isin(months, [6, 7]), 1.15, 1.0))
    covid_dip = np.where(dates.year.to_numpy() == 2020, 0.82, 1.0)
    weekend_boost = np.where(dow >= 5, 1.2, 1.0)
    growth = 1.0 + 0.035 * years_since_2016

    promo_flag = rng.binomial(1, 0.22, size=num_rows)
    discount_pct = np.where(promo_flag == 1, rng.choice([10, 15, 20, 25], size=num_rows), 0)

    base_qty = rng.poisson(lam=3.2, size=num_rows) + 1
    qty = np.maximum(
        1,
        np.round(base_qty * seasonal_boost * covid_dip * weekend_boost * growth * (1 + 0.3 * promo_flag)).astype(int),
    )

    sales = np.round(qty * unit_prices * (1 - discount_pct / 100.0), 2)

    df = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "store_id": chosen_stores,
            "region": chosen_regions,
            "product_id": chosen_products,
            "category": chosen_categories,
            "quantity": qty,
            "unit_price": unit_prices,
            "discount_pct": discount_pct.astype(float),
            "is_promo": promo_flag,
            "sales": sales,
        }
    )

    # Inject realistic data quality issues for Person 1's cleaning pipeline to catch:
    # 1) ~400 missing values in non-critical and critical columns
    df.loc[rng.choice(num_rows, size=250, replace=False), "discount_pct"] = np.nan
    df.loc[rng.choice(num_rows, size=100, replace=False), "region"] = np.nan
    df.loc[rng.choice(num_rows, size=60, replace=False), "sales"] = np.nan

    # 2) ~30 negative/corrupt sales and ~40 extreme typo spikes
    df.loc[rng.choice(num_rows, size=30, replace=False), "sales"] = -150.0
    df.loc[rng.choice(num_rows, size=40, replace=False), "sales"] = 45000.0

    # 3) ~300 exact duplicate rows
    dup_sample = df.sample(n=300, random_state=seed)
    df = pd.concat([df, dup_sample], ignore_index=True)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Generated raw dataset at {OUTPUT_PATH} ({len(df):,} rows, {df.shape[1]} columns)")
    return OUTPUT_PATH


if __name__ == "__main__":
    generate_dataset()
