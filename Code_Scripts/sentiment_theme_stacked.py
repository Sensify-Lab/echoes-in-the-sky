from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/per_theme_stacked_sentiment_2023_05"
)
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
SOURCE_TABLE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_sentiment_changes/tables"
)
PLOT_START = pd.Timestamp("2023-05-01")
SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c0392b",
    "neutral": "#7f8c8d",
    "positive": "#2980b9",
}

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def safe_slug(text: str) -> str:
    text = text.lower().replace("%", "pct")
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text

def style_axis(ax):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def plot_stacked(pivot, ylabel, title, out_stem):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    x_vals = pivot.index.to_pydatetime()
    bottoms = [0.0] * len(pivot)
    for sentiment in SENTIMENT_ORDER:
        vals = pivot[sentiment].values
        ax.bar(
            x_vals,
            vals,
            bottom=bottoms,
            width=18,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            linewidth=0.4,
            label=sentiment.capitalize(),
        )
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    style_axis(ax)
    ax.set_ylabel(ylabel, fontsize=16, fontweight="bold")
    ax.legend(
        frameon=False,
        fontsize=16,
        title_fontsize=16,
        loc="upper left",
        handlelength=1.8,
        borderaxespad=0.8,
    )
    ax.tick_params(axis="both", labelsize=14)
    fig.savefig(FIG_DIR / f"{out_stem}.png", dpi=1020, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{out_stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

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
    counts.to_csv(TABLE_DIR / "theme_sentiment_monthly_counts_from_2023_05.csv", index=False)
    shares.to_csv(TABLE_DIR / "theme_sentiment_monthly_shares_from_2023_05.csv", index=False)
    return counts, shares, theme_order

def build_manifest(theme_order: list[str]) -> pd.DataFrame:
    rows = []
    for i, theme in enumerate(theme_order, start=1):
        slug = safe_slug(theme)
        rows.append(
            {
                "theme_id": f"T{i}",
                "theme": theme,
                "count_png": f"{slug}_stacked_count.png",
                "count_pdf": f"{slug}_stacked_count.pdf",
                "share_png": f"{slug}_stacked_percent.png",
                "share_pdf": f"{slug}_stacked_percent.pdf",
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(TABLE_DIR / "per_theme_stacked_sentiment_manifest.csv", index=False)
    return manifest

def plot_all_themes(counts: pd.DataFrame, shares: pd.DataFrame, theme_order: list[str]) -> None:
    for theme in theme_order:
        slug = safe_slug(theme)
        count_pivot = (
            counts[counts["theme"] == theme]
            .pivot(index="month", columns="sentiment", values="post_count")
            .reindex(columns=SENTIMENT_ORDER)
            .fillna(0)
            .sort_index()
        )
        share_pivot = (
            shares[shares["theme"] == theme]
            .pivot(index="month", columns="sentiment", values="share_pct")
            .reindex(columns=SENTIMENT_ORDER)
            .fillna(0)
            .sort_index()
        )
        plot_stacked(
            count_pivot,
            ylabel="Count (n)",
            title=f"{theme} count",
            out_stem=f"{slug}_stacked_count",
        )
        plot_stacked(
            share_pivot,
            ylabel="Percent (%)",
            title=f"{theme} percent",
            out_stem=f"{slug}_stacked_percent",
        )

def main() -> None:
    ensure_dirs()
    counts, shares, theme_order = load_data()
    build_manifest(theme_order)
    plot_all_themes(counts, shares, theme_order)

if __name__ == "__main__":
    main()
