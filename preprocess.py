"""Data wrangling and preprocessing pipeline for the e-commerce orders dataset.

Stages
------
1. Profile    - measure the quality problems in the raw data
2. Clean      - duplicates, text labels, dates, numeric types, invalid values,
                outliers and missing values
3. Engineer   - create new analytical features
4. Prepare    - encode and scale a model-ready table
5. Report     - save cleaned data, model-ready data and a quality report

Run:  python preprocess.py --input data/raw_orders.csv --output-dir outputs
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("preprocess")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%b %d, %Y"]  # day-first assumed for dd-mm-yyyy
EMAIL_PATTERN = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")

GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}
COUNTRY_MAP = {
    "india": "India",
    "usa": "United States",
    "us": "United States",
    "united states": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "germany": "Germany",
    "australia": "Australia",
}

# Valid inclusive ranges; values outside become missing and are imputed later.
VALID_RANGES = {
    "age": (18, 90),
    "quantity": (1, 50),
    "discount_pct": (0, 90),
    "rating": (1, 5),
    "unit_price": (0.01, np.inf),
}

CATEGORICAL_COLS = ["product_category", "payment_method", "order_status"]
ONE_HOT_COLS = ["product_category", "payment_method", "order_status", "country", "gender", "age_group"]
SCALE_COLS = [
    "age", "quantity", "unit_price", "discount_pct", "rating",
    "shipping_days", "total_amount_log", "customer_order_count", "customer_avg_spend",
]
DROP_FOR_MODEL = ["customer_id", "customer_name", "email", "email_domain",
                  "order_date", "ship_date", "total_amount", "gross_amount", "discount_amount"]


@dataclass
class CleaningStats:
    """Collects how many records each cleaning step touched."""

    steps: dict = field(default_factory=dict)

    def record(self, name: str, count: int) -> None:
        self.steps[name] = int(count)
        log.info("%-38s %6d", name, count)


# --------------------------------------------------------------------------- #
# 1. Profiling
# --------------------------------------------------------------------------- #
def profile(df: pd.DataFrame) -> dict:
    """Summarise size, missingness, duplicates and label consistency."""
    missing = df.isna().sum()
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "missing_cells": int(missing.sum()),
        "missing_pct": round(100 * missing.sum() / df.size, 2),
        "missing_by_column": missing[missing > 0].astype(int).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_order_ids": int(df["order_id"].duplicated().sum()),
        "unique_labels": {c: int(df[c].nunique()) for c in ["gender", "country", "product_category"]},
        "text_columns": [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])],
    }


# --------------------------------------------------------------------------- #
# 2. Cleaning helpers
# --------------------------------------------------------------------------- #
def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)
    return df


def remove_duplicates(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Drop exact duplicates, then keep one row per order_id."""
    before = len(df)
    df = df.drop_duplicates()
    stats.record("exact duplicate rows removed", before - len(df))
    before = len(df)
    df = df.drop_duplicates(subset="order_id", keep="first")
    stats.record("repeated order_id rows removed", before - len(df))
    return df.reset_index(drop=True)


def standardise_labels(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Trim, lower-case and map text labels to one canonical spelling."""
    df = df.copy()
    for col, mapping in (("gender", GENDER_MAP), ("country", COUNTRY_MAP)):
        key = df[col].astype("string").str.strip().str.lower()
        fixed = key.map(mapping)
        stats.record(f"{col} labels standardised", (fixed.notna() & (df[col] != fixed)).sum())
        df[col] = fixed
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("string").str.strip().str.title()
    return df


def parse_mixed_dates(series: pd.Series) -> pd.Series:
    """Parse a text column that mixes several date formats."""
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for fmt in DATE_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(series, format=fmt, errors="coerce"))
    return parsed


def fix_dates(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    df = df.copy()
    for col in ("order_date", "ship_date"):
        df[col] = parse_mixed_dates(df[col])
    swapped = df["ship_date"] < df["order_date"]  # most likely entered in wrong columns
    df.loc[swapped, ["order_date", "ship_date"]] = df.loc[swapped, ["ship_date", "order_date"]].to_numpy()
    stats.record("ship/order dates swapped back", swapped.sum())
    unparseable = df["order_date"].isna()
    stats.record("rows dropped (no valid order_date)", unparseable.sum())
    return df.loc[~unparseable].reset_index(drop=True)


def fix_numeric_types(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Convert text prices such as '$1,200.50' to floats."""
    df = df.copy()
    text_prices = df["unit_price"].astype("string").str.contains(r"[$,]", na=False)
    stats.record("prices converted from text", text_prices.sum())
    cleaned = df["unit_price"].astype("string").str.replace(r"[^0-9.\-]", "", regex=True)
    df["unit_price"] = pd.to_numeric(cleaned, errors="coerce")
    return df


def invalidate_out_of_range(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Turn impossible values (age 250, negative quantity...) into missing values."""
    df = df.copy()
    for col, (low, high) in VALID_RANGES.items():
        invalid = df[col].notna() & ~df[col].between(low, high)
        stats.record(f"invalid {col} set to missing", invalid.sum())
        df[col] = df[col].where(~invalid)
    bad_email = df["email"].notna() & ~df["email"].astype("string").str.match(EMAIL_PATTERN, na=False)
    stats.record("invalid emails set to missing", bad_email.sum())
    df["email"] = df["email"].where(~bad_email)
    return df


def cap_price_outliers(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Winsorise unit_price using the IQR rule inside each product category."""
    df = df.copy()
    grouped = df.groupby("product_category")["unit_price"]
    q1, q3 = grouped.transform(lambda s: s.quantile(0.25)), grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outlier = df["unit_price"].notna() & ((df["unit_price"] < lower) | (df["unit_price"] > upper))
    stats.record("unit_price outliers capped (IQR)", outlier.sum())
    df["price_outlier_flag"] = outlier.astype(int)
    df["unit_price"] = df["unit_price"].clip(lower=lower.clip(lower=0.01), upper=upper)
    return df


def impute_missing(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Fill remaining gaps with rules that fit each column's meaning."""
    df = df.copy()
    for col in ("age", "rating"):  # keep a trace of what was imputed
        df[f"{col}_was_missing"] = df[col].isna().astype(int)

    stats.record("unit_price imputed (category median)", df["unit_price"].isna().sum())
    df["unit_price"] = df["unit_price"].fillna(df.groupby("product_category")["unit_price"].transform("median"))

    for col in ("age", "quantity", "rating"):
        stats.record(f"{col} imputed (median)", df[col].isna().sum())
        df[col] = df[col].fillna(df[col].median())
    df["age"] = df["age"].round().astype(int)
    df["quantity"] = df["quantity"].round().astype(int)

    stats.record("discount_pct imputed (0 = no discount)", df["discount_pct"].isna().sum())
    df["discount_pct"] = df["discount_pct"].fillna(0)

    for col in ("gender", "country"):
        stats.record(f"{col} imputed ('Unknown')", df[col].isna().sum())
        df[col] = df[col].fillna("Unknown")
    return df


def clean(df: pd.DataFrame, stats: CleaningStats) -> pd.DataFrame:
    """Run every cleaning step in a safe order."""
    df = standardise_columns(df)
    df = remove_duplicates(df, stats)
    df = standardise_labels(df, stats)
    df = fix_dates(df, stats)
    df = fix_numeric_types(df, stats)
    df = invalidate_out_of_range(df, stats)
    df = cap_price_outliers(df, stats)
    return impute_missing(df, stats)


# --------------------------------------------------------------------------- #
# 3. Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gross_amount"] = df["quantity"] * df["unit_price"]
    df["discount_amount"] = df["gross_amount"] * df["discount_pct"] / 100
    df["total_amount"] = (df["gross_amount"] - df["discount_amount"]).round(2)

    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_quarter"] = df["order_date"].dt.quarter
    df["order_dayofweek"] = df["order_date"].dt.dayofweek
    df["is_weekend"] = (df["order_dayofweek"] >= 5).astype(int)
    df["shipping_days"] = (df["ship_date"] - df["order_date"]).dt.days

    df["age_group"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 60, 90],
                             labels=["18-25", "26-35", "36-45", "46-60", "60+"]).astype("string")
    df["email_domain"] = df["email"].astype("string").str.split("@").str[1].fillna("unknown")

    by_customer = df.groupby("customer_id")["total_amount"]
    df["customer_order_count"] = by_customer.transform("count")
    df["customer_avg_spend"] = by_customer.transform("mean").round(2)
    df["is_high_value"] = (df["total_amount"] >= df["total_amount"].quantile(0.75)).astype(int)
    df["total_amount_log"] = np.log1p(df["total_amount"])  # reduces right skew
    return df


# --------------------------------------------------------------------------- #
# 4. Encoding and scaling
# --------------------------------------------------------------------------- #
def zscore(series: pd.Series) -> pd.Series:
    std = series.std(ddof=0)
    return (series - series.mean()) / std if std > 0 else series * 0.0


def prepare_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return an all-numeric table: one-hot encoded categories, z-scored numbers."""
    model = df.drop(columns=DROP_FOR_MODEL).copy()
    model["shipping_days"] = model["shipping_days"].fillna(model["shipping_days"].median())
    model[SCALE_COLS] = model[SCALE_COLS].apply(zscore)
    model = pd.get_dummies(model, columns=ONE_HOT_COLS, drop_first=True, dtype=int)
    return model.set_index("order_id")


# --------------------------------------------------------------------------- #
# 5. Orchestration
# --------------------------------------------------------------------------- #
def run(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = standardise_columns(pd.read_csv(input_path))
    before = profile(raw)
    log.info("Loaded %d rows x %d columns", *raw.shape)

    stats = CleaningStats()
    cleaned = engineer_features(clean(raw, stats))
    model_ready = prepare_model_table(cleaned)
    after = profile(cleaned)
    after["new_columns"] = sorted(set(cleaned.columns) - set(raw.columns))

    cleaned.to_csv(output_dir / "cleaned_orders.csv", index=False)
    model_ready.to_csv(output_dir / "model_ready_orders.csv")
    report = {"before": before, "after": after, "cleaning_steps": stats.steps,
              "model_ready_shape": list(model_ready.shape)}
    (output_dir / "quality_report.json").write_text(json.dumps(report, indent=2))

    log.info("Rows: %d -> %d | Missing cells: %d -> %d",
             before["rows"], after["rows"], before["missing_cells"], after["missing_cells"])
    log.info("Model-ready table: %d rows x %d columns", *model_ready.shape)
    log.info("Outputs written to %s", output_dir.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean and prepare the orders dataset.")
    parser.add_argument("--input", type=Path, default=Path("data/raw_orders.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output_dir)
