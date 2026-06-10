from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_sentiment_changes"
)
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
DATA_GLOB = (
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/data_/*.parquet"
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

def build_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    monthly = con.execute(
        f"""
        WITH base AS (
            SELECT
                date_trunc('month', CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE)) AS month,
                Theme_Name AS theme,
                "record.text.clean.sentimentlabel" AS sentiment
            FROM read_parquet('{DATA_GLOB}')
            WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
              AND Theme_Name IS NOT NULL
              AND "record.text.clean.sentimentlabel" IS NOT NULL
        )
        SELECT month, theme, sentiment, count(*) AS post_count
        FROM base
        GROUP BY 1, 2, 3
        ORDER BY 1, 2, 3
        """
    ).df()
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly.to_csv(TABLE_DIR / "theme_sentiment_monthly_counts.csv", index=False)

    monthly_totals = (
        monthly.groupby(["month", "theme"], as_index=False)["post_count"]
        .sum()
        .rename(columns={"post_count": "theme_month_total"})
    )
    shares = monthly.merge(monthly_totals, on=["month", "theme"], how="left")
    shares["share_pct"] = shares["post_count"] / shares["theme_month_total"] * 100.0
    shares.to_csv(TABLE_DIR / "theme_sentiment_monthly_shares.csv", index=False)

    theme_summary = (
        monthly.groupby(["theme", "sentiment"], as_index=False)["post_count"]
        .sum()
        .sort_values(["theme", "sentiment"])
    )
    theme_summary.to_csv(TABLE_DIR / "theme_sentiment_overall_counts.csv", index=False)
    return monthly, shares, theme_summary

def get_theme_order(theme_summary: pd.DataFrame) -> list[str]:
    totals = (
        theme_summary.groupby("theme", as_index=False)["post_count"]
        .sum()
        .sort_values("post_count", ascending=False)
    )
    return totals["theme"].tolist()

def style_month_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def plot_negative_heatmap(shares: pd.DataFrame, theme_order: list[str]) -> None:
    neg = shares[
        (shares["sentiment"] == "negative") & (shares["month"] >= PLOT_START)
    ].copy()
    pivot = neg.pivot(index="theme", columns="month", values="share_pct").reindex(theme_order)
    fig, ax = plt.subplots(figsize=(18, max(8, len(theme_order) * 0.6)))
    im = ax.imshow(pivot.fillna(np.nan).values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    month_labels = [pd.to_datetime(x).strftime("%Y-%m") for x in pivot.columns]
    keep = [i for i in range(len(month_labels)) if i % 3 == 0 or i == len(month_labels) - 1]
    ax.set_xticks(keep)
    ax.set_xticklabels([month_labels[i] for i in keep], rotation=45, ha="right", fontsize=10)
    ax.set_title("Monthly Negative Sentiment Share by Theme", fontsize=18, fontweight="bold", pad=14)
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Negative share (%)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theme_negative_share_heatmap.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "theme_negative_share_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_theme_stacked_bar(theme_summary: pd.DataFrame, theme_order: list[str]) -> None:
    pivot = (
        theme_summary.pivot(index="theme", columns="sentiment", values="post_count")
        .reindex(index=theme_order, columns=SENTIMENT_ORDER)
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(16, max(8, len(theme_order) * 0.55)))
    left = np.zeros(len(pivot))
    for sentiment in SENTIMENT_ORDER:
        vals = pivot[sentiment].values
        ax.barh(
            pivot.index,
            vals,
            left=left,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            linewidth=0.5,
            label=sentiment.capitalize(),
        )
        left += vals
    ax.invert_yaxis()
    ax.set_title("Overall Post Volume by Theme and Sentiment", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Posts", fontsize=13)
    ax.legend(frameon=False, fontsize=11, title="Sentiment", title_fontsize=11)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theme_sentiment_overall_stacked_barh.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "theme_sentiment_overall_stacked_barh.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_small_multiples(shares: pd.DataFrame, theme_order: list[str]) -> None:
    shares = shares[shares["month"] >= PLOT_START].copy()
    n = len(theme_order)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3.4), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)
    for ax, theme in zip(axes, theme_order):
        df = shares[shares["theme"] == theme]
        pivot = (
            df.pivot(index="month", columns="sentiment", values="share_pct")
            .reindex(columns=SENTIMENT_ORDER)
            .fillna(0)
            .sort_index()
        )
        for sentiment in SENTIMENT_ORDER:
            ax.plot(
                pivot.index,
                pivot[sentiment],
                color=SENTIMENT_COLORS[sentiment],
                linewidth=2.2,
                label=sentiment.capitalize(),
            )
        ax.set_title(theme, fontsize=11, loc="left")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.15, linestyle="--")
        style_month_axis(ax)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Monthly Sentiment Composition by Theme", fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(FIG_DIR / "theme_sentiment_small_multiples.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "theme_sentiment_small_multiples.pdf", bbox_inches="tight")
    plt.close(fig)

def plot_negative_change( shares: pd.DataFrame, theme_order: list[str]) -> None:
    neg = shares[shares["sentiment"] == "negative"].copy().sort_values(["theme", "month"])
    summary = (
        neg.groupby("theme", as_index=False)
        .agg(first_share=("share_pct", "first"), last_share=("share_pct", "last"))
    )
    summary["change_pp"] = summary["last_share"] - summary["first_share"]
    summary = summary.set_index("theme").loc[theme_order].reset_index()
    fig, ax = plt.subplots(figsize=(14, max(8, len(summary) * 0.5)))
    colors = ["#c0392b" if x >= 0 else "#2980b9" for x in summary["change_pp"]]
    ax.barh(summary["theme"], summary["change_pp"], color=colors)
    ax.axvline(0, color="black", linewidth=1)
    ax.invert_yaxis()
    ax.set_title("Change in Negative Sentiment Share from First to Last Month", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Change in negative share (percentage points)", fontsize=13)
    ax.grid(axis="x", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "theme_negative_share_change_barh.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "theme_negative_share_change_barh.pdf", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    monthly, shares, theme_summary = build_tables()
    theme_order = get_theme_order(theme_summary)
    plot_negative_heatmap(shares, theme_order)
    plot_theme_stacked_bar(theme_summary, theme_order)
    plot_small_multiples(shares, theme_order)
    plot_negative_change(shares, theme_order)

if __name__ == "__main__":
    main()
