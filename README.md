# Data Wrangling and Preprocessing Strategy: E-commerce Orders

A reproducible pandas/NumPy pipeline that turns a deliberately messy orders
dataset into an analysis-ready and a model-ready table.

## Project structure

| File | Purpose |
|---|---|
| `generate_data.py` | Builds a seeded synthetic dataset (5,200 rows) with injected quality problems |
| `preprocess.py` | Profiling, cleaning, feature engineering, encoding, scaling, reporting |
| `visualize.py` | Before/after charts used in the report |
| `requirements.txt` | Python dependencies |
| `data/raw_orders.csv` | Raw, messy input |
| `outputs/cleaned_orders.csv` | Clean table with engineered features (for EDA) |
| `outputs/model_ready_orders.csv` | One-hot encoded and z-scored table (for ML) |
| `outputs/quality_report.json` | Before/after quality metrics and per-step counts |

## Run

```bash
pip install -r requirements.txt
python generate_data.py     # creates data/raw_orders.csv
python preprocess.py        # cleans, engineers features, writes outputs/
python visualize.py         # writes outputs/figures/*.png
```

## Pipeline

1. **Profile**: missing values, duplicates, label variants, text columns
2. **Clean**: remove duplicates, standardise labels, parse mixed date formats,
   convert text prices, invalidate impossible values, cap outliers (IQR per
   category), impute missing values with column-appropriate rules
3. **Engineer**: `total_amount`, date parts, `shipping_days`, `age_group`,
   customer-level aggregates, `is_high_value`, `log1p` of amount
4. **Prepare**: drop identifiers, one-hot encode categories, z-score numerics
5. **Report**: save cleaned data, model-ready data and a JSON quality report

## Key assumptions

- `dd-mm-yyyy` dates are day-first.
- A ship date earlier than the order date means the two were entered in the wrong columns, so they are swapped.
- A missing discount means no discount; other numerics use the median (price uses the category median).
- Emails are not imputed; they are only used to derive `email_domain` and are excluded from the model table.
