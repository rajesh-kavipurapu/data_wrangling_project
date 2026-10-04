"""Generate a deliberately messy e-commerce orders dataset.

The data is synthetic and seeded, so every run produces the same file.
Each quality problem injected below is later fixed by ``preprocess.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ROWS = 5000
OUTPUT_PATH = Path("data/raw_orders.csv")

CATEGORIES = ["Electronics", "Clothing", "Home & Kitchen", "Books", "Sports", "Beauty"]
PRICE_RANGE = {  # (low, high) typical unit price per category
    "Electronics": (80, 900),
    "Clothing": (15, 120),
    "Home & Kitchen": (10, 250),
    "Books": (5, 40),
    "Sports": (15, 300),
    "Beauty": (5, 90),
}
COUNTRIES = ["India", "United States", "United Kingdom", "Germany", "Australia"]
PAYMENTS = ["Credit Card", "Debit Card", "UPI", "PayPal", "Cash on Delivery"]
STATUSES = ["Delivered", "Shipped", "Cancelled", "Returned"]
DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%b %d, %Y"]


def build_clean_orders(rng: np.random.Generator) -> pd.DataFrame:
    """Create a tidy 'ground truth' table before any problems are injected."""
    n = N_ROWS
    category = rng.choice(CATEGORIES, n, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.1])
    low = np.array([PRICE_RANGE[c][0] for c in category])
    high = np.array([PRICE_RANGE[c][1] for c in category])
    order_date = pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 640, n), unit="D")

    customer_num = rng.integers(1, 1201, n)
    return pd.DataFrame(
        {
            "order_id": [f"ORD{100000 + i}" for i in range(n)],
            "customer_id": [f"CUST{c:04d}" for c in customer_num],
            "customer_name": [f"Customer {c}" for c in customer_num],
            "email": [f"user{c}@{rng.choice(['gmail.com', 'yahoo.com', 'outlook.com'])}" for c in customer_num],
            "age": np.clip(rng.normal(36, 11, n).round(), 18, 80),
            "gender": rng.choice(["Male", "Female"], n),
            "country": rng.choice(COUNTRIES, n, p=[0.4, 0.25, 0.15, 0.1, 0.1]),
            "order_date": order_date,
            "ship_date": order_date + pd.to_timedelta(rng.integers(1, 9, n), unit="D"),
            "product_category": category,
            "quantity": rng.integers(1, 6, n).astype(float),
            "unit_price": rng.uniform(low, high).round(2),
            "discount_pct": rng.choice([0, 5, 10, 15, 20], n, p=[0.4, 0.2, 0.2, 0.1, 0.1]).astype(float),
            "payment_method": rng.choice(PAYMENTS, n),
            "order_status": rng.choice(STATUSES, n, p=[0.7, 0.15, 0.1, 0.05]),
            "rating": rng.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3]).astype(float),
        }
    )


def pick(rng: np.random.Generator, n: int, fraction: float) -> np.ndarray:
    """Return random row positions covering roughly ``fraction`` of the rows."""
    return rng.choice(n, int(n * fraction), replace=False)


def inject_problems(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add realistic data-quality issues to a clean table."""
    df = df.copy()
    n = len(df)

    # 1. Missing values (different rates per column).
    for col, frac in {"age": 0.08, "email": 0.05, "country": 0.04,
                      "discount_pct": 0.06, "rating": 0.12, "unit_price": 0.03}.items():
        df.loc[pick(rng, n, frac), col] = np.nan

    # 2. Inconsistent category labels (case, spaces, abbreviations).
    gender_idx = pick(rng, n, 0.30)
    df["gender"] = df["gender"].astype(object)
    df.loc[gender_idx, "gender"] = rng.choice(["M", "F", "male", "FEMALE", " Male ", "f"], len(gender_idx))
    country_idx = pick(rng, n, 0.15)
    df["country"] = df["country"].astype(object)
    df.loc[country_idx, "country"] = rng.choice(["india", "INDIA", "USA", "US", "uk", " Germany"], len(country_idx))
    df["product_category"] = df["product_category"].astype(object)
    cat_idx = pick(rng, n, 0.10)
    df.loc[cat_idx, "product_category"] = df.loc[cat_idx, "product_category"].str.lower()

    # 3. Dates stored in three different text formats.
    for col in ("order_date", "ship_date"):
        fmt_choice = rng.integers(0, len(DATE_FORMATS), n)
        df[col] = [d.strftime(DATE_FORMATS[f]) for d, f in zip(df[col], fmt_choice)]

    # 4. Prices stored as text with currency symbols or thousands separators.
    price_idx = pick(rng, n, 0.10)
    price_text = df["unit_price"].astype(object)
    price_text.loc[price_idx] = [f"${p:,.2f}" if pd.notna(p) else p for p in df.loc[price_idx, "unit_price"]]
    df["unit_price"] = price_text

    # 5. Impossible or extreme values.
    df.loc[pick(rng, n, 0.01), "age"] = rng.choice([-5, 0, 150, 250])
    df.loc[pick(rng, n, 0.01), "quantity"] = rng.choice([-1, 0, -3])
    df.loc[pick(rng, n, 0.005), "discount_pct"] = rng.choice([150, -10])
    df.loc[pick(rng, n, 0.01), "rating"] = rng.choice([0, 7, 10])
    outlier_idx = pick(rng, n, 0.008)
    df.loc[outlier_idx, "unit_price"] = rng.choice([25000, 48000, 99999], len(outlier_idx))

    # 6. Malformed emails.
    bad_email_idx = pick(rng, n, 0.03)
    df.loc[bad_email_idx, "email"] = rng.choice(["not_an_email", "user@@mail", "john.doe.com"], len(bad_email_idx))

    # 7. Logical inconsistency: ship date earlier than order date.
    swap_idx = pick(rng, n, 0.02)
    df.loc[swap_idx, ["order_date", "ship_date"]] = df.loc[swap_idx, ["ship_date", "order_date"]].to_numpy()

    # 8. Exact duplicate rows and repeated order IDs with changed values.
    exact_dupes = df.iloc[pick(rng, n, 0.03)]
    changed_dupes = df.iloc[pick(rng, n, 0.01)].copy()
    changed_dupes["quantity"] = changed_dupes["quantity"].abs() + 1
    df = pd.concat([df, exact_dupes, changed_dupes], ignore_index=True)

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(SEED)
    raw = inject_problems(build_clean_orders(rng), rng)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(raw):,} rows x {raw.shape[1]} columns to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
