# Dataset Documentation (Person 1)

## 1. Dataset Overview & Source
- **Active Local Dataset:** `data/raw/sales.csv` (120,000+ rows covering `2016-01-01` to `2025-12-31`, generated via `scripts/generate_sample_data.py` so the full cleaning, EDA, and ML pipeline runs out-of-the-box offline).
- **Recommended External Kaggle Dataset (if swapping to an external public source):**
  - *India Q-Commerce Orders Dataset (2015–2025)* — `https://www.kaggle.com/datasets/ahmedabid/india-q-commerce-orders-dataset-2015-2025`
  - To switch to any Kaggle CSV, place it at `data/raw/sales.csv`, update the 4 column-name constants at the top of `preprocessing/data_cleaning.py` and `preprocessing/eda.py`, and re-run both scripts.

## 2. Dataset Size
- **Raw Dataset:** `120,300` rows, `10` columns
- **Cleaned Dataset:** `119,650` rows, `10` columns
- **Rows/Values Cleaned:**
  - `560` duplicate rows removed
  - `60` rows dropped for missing critical fields (`sales`)
  - `350` non-critical missing cells imputed (`discount_pct` with training-period median, `region` with `"Unknown"`)
  - `30` invalid negative-sales rows removed
  - `672` extreme upper outliers capped at `3,968.82` (99.5th percentile of training data)

## 3. Column Descriptions
| Column Name | Data Type | Meaning |
| :--- | :--- | :--- |
| `date` | datetime64 | Transaction date (`2016-01-01` to `2025-12-31`) |
| `store_id` | string | Store identifier (`STORE_01` – `STORE_15`) |
| `region` | string | Store geographical region |
| `product_id` | string | Product identifier (`PROD_001` – `PROD_050`) |
| `category` | string | Product category (`Electronics`, `Groceries`, `Apparel`, `Home & Kitchen`, `Personal Care`) |
| `quantity` | int64 | Units sold in the transaction |
| `unit_price` | float64 | Price per unit |
| `discount_pct` | float64 | Discount percentage applied (`0`–`25`) |
| `is_promo` | int64 | Promotional flag (`1` = promo active, `0` = no promo) |
| `sales` | float64 | **Target variable** — total sales revenue |

## 4. Data Cleaning & Missing-Value Handling
1. **Date Parsing:** Converted `date` to `datetime64`, dropped unparseable records, and filtered strictly to `2016-01-01` through `2025-12-31`.
2. **Duplicates:** Detected and removed exact duplicate rows.
3. **Missing Values:** Dropped rows missing critical keys (`date`, `store_id`, `product_id`, `sales`); imputed missing numeric columns (`discount_pct`) using the **training-period median** (to prevent future data leakage) and missing text columns (`region`) with `"Unknown"`.
4. **Outliers:** Removed impossible negative sales records and capped extreme upper spikes at the **99.5th percentile computed on the training period**.

## 5. Chronological Train / Test Split (8 Years / 2 Years)
- **Training Split:** `2016-01-01` to `2023-12-31` (8 years) → `95,788` rows (`80.1%`) → `data/processed/train_sales.csv`
- **Testing Split:** `2024-01-01` to `2025-12-31` (2 years) → `23,862` rows (`19.9%`) → `data/processed/test_sales.csv`
- Split is strictly chronological (`max(train.date) < min(test.date)`) to prevent temporal data leakage.

## 6. How to Reproduce
```powershell
python scripts/generate_sample_data.py
python preprocessing/data_cleaning.py
python preprocessing/eda.py
pytest tests/test_preprocessing.py
```
