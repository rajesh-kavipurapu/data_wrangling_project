"""Create before/after charts that show the effect of preprocessing."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocess import fix_numeric_types, CleaningStats, standardise_columns

BEFORE_COLOR, AFTER_COLOR = "#C0504D", "#2E75B6"


def plot_missing(raw: pd.DataFrame, cleaned: pd.DataFrame, path: Path) -> None:
    cols = ["age", "email", "country", "discount_pct", "rating", "unit_price"]
    x, width = np.arange(len(cols)), 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, raw[cols].isna().sum(), width, label="Before", color=BEFORE_COLOR)
    ax.bar(x + width / 2, cleaned[cols].isna().sum(), width, label="After", color=AFTER_COLOR)
    ax.set_xticks(x, cols, rotation=20)
    ax.set_ylabel("Missing values")
    ax.set_title("Missing values per column")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_price_outliers(raw: pd.DataFrame, cleaned: pd.DataFrame, path: Path) -> None:
    raw_price = fix_numeric_types(raw, CleaningStats())["unit_price"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, data, title, color in ((axes[0], raw_price, "Before (with outliers)", BEFORE_COLOR),
                                   (axes[1], cleaned["unit_price"], "After (IQR capping)", AFTER_COLOR)):
        ax.boxplot(data, patch_artist=True, boxprops={"facecolor": color, "alpha": 0.6})
        ax.set_title(title)
        ax.set_ylabel("unit_price")
        ax.set_xticks([])
    fig.suptitle("Effect of outlier handling on unit_price")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_skew(cleaned: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].hist(cleaned["total_amount"], bins=40, color=BEFORE_COLOR)
    axes[0].set_title(f"total_amount (skew {cleaned['total_amount'].skew():.2f})")
    axes[1].hist(cleaned["total_amount_log"], bins=40, color=AFTER_COLOR)
    axes[1].set_title(f"log1p(total_amount) (skew {cleaned['total_amount_log'].skew():.2f})")
    fig.suptitle("Log transform reduces right skew")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/raw_orders.csv"))
    parser.add_argument("--cleaned", type=Path, default=Path("outputs/cleaned_orders.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw = standardise_columns(pd.read_csv(args.raw))
    cleaned = pd.read_csv(args.cleaned)
    plot_missing(raw, cleaned, args.output_dir / "missing_values.png")
    plot_price_outliers(raw, cleaned, args.output_dir / "price_outliers.png")
    plot_skew(cleaned, args.output_dir / "amount_skew.png")
    print(f"Figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
