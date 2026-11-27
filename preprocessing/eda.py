"""Exploratory Data Analysis (EDA) graph generator for Sales Predictor (Person 1).

Generates the 5 required EDA graphs into docs/screenshots/:
1. Sales trends over time (monthly)
2. Product-wise sales (top 10 products)
3. Store-wise sales (all / top 15 stores)
4. Sales distribution (histogram + KDE)
5. Correlation heatmap
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reliable script execution
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_sales.csv"
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"

DATE_COL = "date"
STORE_COL = "store_id"
PRODUCT_COL = "product_id"
SALES_COL = "sales"


def generate_eda_plots(
    clean_path: Path = CLEAN_DATA_PATH,
    output_dir: Path = SCREENSHOTS_DIR,
) -> list[Path]:
    """Generate and save all 5 required EDA charts."""
    if not clean_path.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found at {clean_path}. Run preprocessing/data_cleaning.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(clean_path, parse_dates=[DATE_COL])
    sns.set_theme(style="whitegrid")

    saved_files: list[Path] = []

    # 1. Sales trends over time (Monthly)
    monthly = df.set_index(DATE_COL)[SALES_COL].resample("ME").sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly.index, monthly.values, color="#2563eb", linewidth=2)
    ax.axvline(pd.Timestamp("2024-01-01"), color="#dc2626", linestyle="--", label="Train/Test Cutoff (2024-01-01)")
    ax.set_title("Monthly Sales Trend (2016–2025)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    ax.legend()
    fig.tight_layout()
    p1 = output_dir / "01_sales_trend_over_time.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    saved_files.append(p1)

    # 2. Product-wise sales (Top 10)
    top_products = df.groupby(PRODUCT_COL)[SALES_COL].sum().nlargest(10).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    top_products.plot(kind="barh", color="#059669", ax=ax)
    ax.set_title("Top 10 Products by Total Sales", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Product")
    fig.tight_layout()
    p2 = output_dir / "02_product_wise_sales.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    saved_files.append(p2)

    # 3. Store-wise sales (Top 15)
    store_sales = df.groupby(STORE_COL)[SALES_COL].sum().nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    store_sales.plot(kind="barh", color="#d97706", ax=ax)
    ax.set_title("Store-wise Total Sales (Top 15 Stores)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Total Sales")
    ax.set_ylabel("Store")
    fig.tight_layout()
    p3 = output_dir / "03_store_wise_sales.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    saved_files.append(p3)

    # 4. Sales distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[SALES_COL], bins=50, kde=True, color="#7c3aed", ax=ax)
    ax.set_title("Sales Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sales")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    p4 = output_dir / "04_sales_distribution.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    saved_files.append(p4)

    # 5. Correlation heatmap
    num_df = df.select_dtypes(include="number").copy()
    num_df["year"] = df[DATE_COL].dt.year
    num_df["month"] = df[DATE_COL].dt.month
    num_df["day_of_week"] = df[DATE_COL].dt.day_of_week
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p5 = output_dir / "05_correlation_heatmap.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    saved_files.append(p5)

    print("Generated EDA graphs:")
    for path in saved_files:
        print(f" - {path}")
    return saved_files


if __name__ == "__main__":
    generate_eda_plots()
