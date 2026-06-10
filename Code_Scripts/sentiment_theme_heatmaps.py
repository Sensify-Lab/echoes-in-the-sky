from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_month_heatmaps_2023_06"
)
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
SOURCE_TABLE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_sentiment_changes/tables"
)
PLOT_START = pd.Timestamp("2023-06-01")
SENTIMENT_ORDER = ["negative", "neutral", "positive"]

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    counts = pd.read_csv(SOURCE_TABLE_DIR / "theme_sentiment_monthly_counts.csv", parse_dates=["month"])
    shares = pd.read_csv(SOURCE_TABLE_DIR / "theme_sentiment_monthly_shares.csv", parse_dates=["month"])
    counts = counts[counts["month"] >= PLOT_START].copy()
    shares = shares[shares["month"] >= PLOT_START].copy()
    theme_order = (
        counts.groupby("theme", as_index=False)["post_count"]
        .sum()
        .sort_values("post_count", ascending=False)["theme"]
        .tolist()
    )
    counts.to_csv(TABLE_DIR / "theme_month_counts_from_2023_06.csv", index=False)
    shares.to_csv(TABLE_DIR / "theme_month_shares_from_2023_06.csv", index=False)
    return counts, shares, theme_order

def month_labels(months: list[pd.Timestamp]) -> list[str]:
    return [pd.to_datetime(m).strftime("%Y-%m") for m in months]

def plot_count_heatmap(counts: pd.DataFrame, theme_order: list[str]) -> None:
    totals = counts.groupby(["theme", "month"], as_index=False)["post_count"].sum()
    month_totals = totals.groupby("month", as_index=False)["post_count"].sum().rename(
        columns={"post_count": "month_total"}
    )
    totals = totals.merge(month_totals, on="month", how="left")
    totals["month_pct"] = totals["post_count"] / totals["month_total"] * 100.0

    value_pivot = totals.pivot(index="theme", columns="month", values="post_count").reindex(theme_order)
    pct_pivot = totals.pivot(index="theme", columns="month", values="month_pct").reindex(theme_order)

    months = list(value_pivot.columns)
    fig, ax = plt.subplots(figsize=(16, max(7, len(theme_order) * 0.48)))
    im = ax.imshow(value_pivot.values, aspect="auto", cmap="YlGnBu")

    ax.set_yticks(range(len(theme_order)))
    ax.set_yticklabels(theme_order, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels(months), rotation=45, ha="right", fontsize=11, fontweight="bold")
    ax.set_title("Monthly Bluesky Post Counts by Theme", fontsize=21, fontweight="bold", pad=12)

    for i in range(value_pivot.shape[0]):
        for j in range(value_pivot.shape[1]):
            val = value_pivot.iat[i, j]
            pct = pct_pivot.iat[i, j]
            if pd.isna(val):
                continue
            txt = f"{pct:.0f}%"
            color = "white" if val >= np.nanmax(value_pivot.values) * 0.45 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Monthly post count", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "theme_month_count_heatmap_pct_annotated.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "theme_month_count_heatmap_pct_annotated.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_sentiment_heatmap(counts: pd.DataFrame, shares: pd.DataFrame, theme_order: list[str]) -> None:
    total_counts = counts.groupby(["theme", "month"], as_index=False)["post_count"].sum().rename(
        columns={"post_count": "month_posts"}
    )
    wide = (
        shares.pivot(index=["theme", "month"], columns="sentiment", values="share_pct")
        .reindex(columns=SENTIMENT_ORDER)
        .fillna(0)
        .reset_index()
    )
    wide = wide.merge(total_counts, on=["theme", "month"], how="left")
    wide["sentiment_balance"] = wide["positive"] - wide["negative"]
    wide["dominant_sentiment"] = wide[SENTIMENT_ORDER].idxmax(axis=1)

    balance_pivot = wide.pivot(index="theme", columns="month", values="sentiment_balance").reindex(theme_order)
    dom_pivot = wide.pivot(index="theme", columns="month", values="dominant_sentiment").reindex(theme_order)
    months = list(balance_pivot.columns)

    fig, ax = plt.subplots(figsize=(16, max(7, len(theme_order) * 0.48)))
    im = ax.imshow(balance_pivot.values, aspect="auto", cmap="RdBu", vmin=-100, vmax=100)

    ax.set_yticks(range(len(theme_order)))
    ax.set_yticklabels(theme_order, fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(month_labels(months), rotation=45, ha="right", fontsize=11, fontweight="bold")
    ax.set_title(
        "Monthly Theme Sentiment Balance (Positive Share - Negative Share)",
        fontsize=21,
        fontweight="bold",
        pad=12,
    )

    initials = {"negative": "N", "neutral": "U", "positive": "P"}
    for i in range(balance_pivot.shape[0]):
        for j in range(balance_pivot.shape[1]):
            val = balance_pivot.iat[i, j]
            dom = dom_pivot.iat[i, j]
            if pd.isna(val):
                continue
            txt = f"{initials.get(dom, '?')} {val:+.0f}"
            color = "white" if abs(val) >= 35 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=color, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Sentiment balance (percentage points)", fontsize=13, fontweight="bold")
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG_DIR / "theme_month_sentiment_balance_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "theme_month_sentiment_balance_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    wide.to_csv(TABLE_DIR / "theme_month_sentiment_balance_table.csv", index=False)

def main() -> None:
    ensure_dirs()
    counts, shares, theme_order = load_data()
    plot_count_heatmap(counts, theme_order)
    plot_sentiment_heatmap(counts, shares, theme_order)

if __name__ == "__main__":
    main()
