from __future__ import annotations

from pathlib import Path
import json
import shutil

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SOURCE_TABLE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_sentiment_changes/tables"
)
BASE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/"
    "OVERLEAF_monthly_text_sentiment_overview_overleaf_figure_may22_per_theme"
)
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
COUNTS_FIG_DIR = FIG_DIR / "counts"
SHARES_FIG_DIR = FIG_DIR / "shares"

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c0392b",
    "neutral": "#7f8c8d",
    "positive": "#2980b9",
}

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    COUNTS_FIG_DIR.mkdir(parents=True, exist_ok=True)
    SHARES_FIG_DIR.mkdir(parents=True, exist_ok=True)

def style_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def plot_stacked(pivot: pd.DataFrame, theme: str, fig_path: Path) -> None:
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
    ax.set_ylabel(
        "Count (n)",
        fontsize=16,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        fontsize=16,
        title_fontsize=16,
        loc="upper left",
        handlelength=1.8,
        borderaxespad=0.8,
    )
    ax.tick_params(axis="both", labelsize=14)

    fig.savefig(
        fig_path,
        dpi=1020,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        fig_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

def plot_stacked_share(pivot: pd.DataFrame, theme: str, fig_path: Path) -> None:
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
    ax.tick_params(axis="both", labelsize=14)
    ax.set_ylabel(
        "Percentage (%)",
        fontsize=16,
        fontweight="bold",
    )

    fig.savefig(
        fig_path,
        dpi=1020,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        fig_path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    counts = pd.read_csv(SOURCE_TABLE_DIR / "theme_sentiment_monthly_counts.csv", parse_dates=["month"])
    shares = pd.read_csv(SOURCE_TABLE_DIR / "theme_sentiment_monthly_shares.csv", parse_dates=["month"])

    counts = counts.loc[counts["month"] >= "2023-06-01"].copy()
    shares = shares.loc[shares["month"] >= "2023-06-01"].copy()

    counts.to_csv(TABLE_DIR / "theme_monthly_text_sentiment_counts.csv", index=False)
    shares.to_csv(TABLE_DIR / "theme_monthly_text_sentiment_shares.csv", index=False)

    summary = (
        counts.groupby("theme", as_index=False)
        .agg(
            n_months=("month", "nunique"),
            series_start=("month", "min"),
            series_end=("month", "max"),
            total_posts=("post_count", "sum"),
        )
        .sort_values("theme")
    )
    summary.to_csv(TABLE_DIR / "theme_monthly_text_sentiment_summary.csv", index=False)
    return counts, shares, summary

def build_figures(counts: pd.DataFrame, shares: pd.DataFrame) -> None:
    for theme in sorted(counts["theme"].dropna().unique()):
        theme_counts = counts.loc[counts["theme"] == theme]
        theme_shares = shares.loc[shares["theme"] == theme]

        count_pivot = (
            theme_counts.pivot(index="month", columns="sentiment", values="post_count")
            .reindex(columns=SENTIMENT_ORDER)
            .fillna(0)
        )
        share_pivot = (
            theme_shares.pivot(index="month", columns="sentiment", values="share_pct")
            .reindex(columns=SENTIMENT_ORDER)
            .fillna(0)
        )

        stem = slugify(theme)
        plot_stacked(
            count_pivot,
            theme,
            COUNTS_FIG_DIR / f"{stem}_monthly_bluesky_posts_by_text_sentiment_label.png",
        )
        plot_stacked_share(
            share_pivot,
            theme,
            SHARES_FIG_DIR / f"{stem}_monthly_bluesky_sentiment_composition_by_text_sentiment_label.png",
        )

def main() -> None:
    ensure_dirs()
    counts, shares, _summary = build_tables()
    build_figures(counts, shares)

if __name__ == "__main__":
    main()
