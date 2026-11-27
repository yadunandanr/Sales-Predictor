# LLD Handover Notes – Person 1 (Data, EDA & Preprocessing)

Send the contents below + the 5 PNGs in `docs/screenshots/` to **Person 5** for the LLD report, and share Section 3 (Schema) with **Person 2** and **Person 4**.

---

## 1. Dataset Source & Description
- **Dataset Source:** Locally generated realistic retail transaction dataset (`scripts/generate_sample_data.py`), designed for offline reproducibility across `2016–2025`. *(Optional external Kaggle equivalent if swapped before final submission: India Q-Commerce Orders Dataset 2015–2025 — `https://www.kaggle.com/datasets/ahmedabid/india-q-commerce-orders-dataset-2015-2025`.)*
- **Description:** Multi-store, multi-product daily retail sales dataset covering **10 years (`2016-01-01` to `2025-12-31`)** across 15 stores, 5 geographical regions, and 50 products across 5 categories (`Electronics`, `Groceries`, `Apparel`, `Home & Kitchen`, `Personal Care`).
- **Raw Dimensions:** `120,300` rows × `10` columns (`> 1 lakh` records).
- **Cleaned Dimensions:** `119,650` rows × `10` columns.

## 2. Schema (Columns & Data Types)
| Column | Data Type | Role / Meaning |
| :--- | :--- | :--- |
| `date` | `datetime64` | Transaction date (`2016-01-01` to `2025-12-31`) |
| `store_id` | `string` | Unique store ID (`STORE_01` – `STORE_15`) |
| `region` | `string` | Store region (`North`, `South`, `East`, `West`, `Central`) |
| `product_id` | `string` | Unique product ID (`PROD_001` – `PROD_050`) |
| `category` | `string` | Product category |
| `quantity` | `int64` | Number of units sold |
| `unit_price` | `float64` | Price per unit |
| `discount_pct` | `float64` | Promotional discount percentage (`0`–`25`) |
| `is_promo` | `int64` | Binary promo indicator (`1` = promo, `0` = normal) |
| `sales` | `float64` | **Target variable** — total transaction sales |

## 3. Data Cleaning & Missing-Value Handling
- **Date formatting:** Parsed `date` to `datetime64` (`0` invalid dates, `0` out-of-range dates).
- **Duplicates:** Removed `560` exact duplicate rows.
- **Missing values:**
  - Dropped `60` rows missing the critical target column (`sales`).
  - Imputed `350` non-critical missing cells (`discount_pct` using the training-period median to avoid future data leakage; `region` with `"Unknown"`).
- **Outliers:**
  - Removed `30` rows with invalid negative sales.
  - Capped `672` extreme high outliers at `3,968.82` (the `99.5th` percentile calculated strictly on the training period).

## 4. Chronological Train / Test Split
- **Training Set (8 years):** `2016-01-01` to `2023-12-31` → **`95,788` rows (`80.1%`)**
- **Testing Set (2 years):** `2024-01-01` to `2025-12-31` → **`23,862` rows (`19.9%`)**
- Strictly chronological (`max(train.date) < min(test.date)`) to prevent future sales leakage.

## 5. EDA Graphs (Attached in `docs/screenshots/`)
1. `01_sales_trend_over_time.png` — Monthly sales trend (`2016–2025`), showing yearly growth, November–December holiday spikes, a 2020 dip, and the `2024-01-01` train/test cutoff.
2. `02_product_wise_sales.png` — Top 10 products ranked by total revenue.
3. `03_store_wise_sales.png` — Store-wise total sales across all 15 stores.
4. `04_sales_distribution.png` — Right-skewed transaction sales distribution with KDE curve.
5. `05_correlation_heatmap.png` — Correlation matrix among numeric features (`quantity`, `unit_price`, `discount_pct`, `is_promo`, `sales`, `year`, `month`, `day_of_week`).

## 6. Dataset Limitations (For LLD Section 6)
- Synthetic retail transactions lack external macroeconomic signals (weather, local holidays, competitor pricing, stock-outs).
- Transaction sales are right-skewed, so tree-based models (XGBoost) may outperform Linear Regression unless Person 2 applies scaling/feature transformations.
- Does not include product cost/profit margin or customer-level loyalty history.

## 7. Unit Test Cases (For LLD Section 5)
All 5 tests in `tests/test_preprocessing.py` pass:
1. `test_parse_and_filter_dates_drops_invalid_and_out_of_range` — Verifies invalid and out-of-window dates are excluded and `date` becomes `datetime64`.
2. `test_remove_duplicates` — Verifies duplicate rows are detected and removed.
3. `test_handle_missing_values_drops_critical_and_imputes_non_critical` — Verifies critical nulls are dropped and remaining nulls are imputed with zero leftover NaNs.
4. `test_handle_outliers_removes_negative_and_caps_extreme` — Verifies negative sales are removed and high outliers are capped.
5. `test_chronological_split_has_zero_leakage` — Verifies `max(train.date) < min(test.date)`.
