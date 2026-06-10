from __future__ import annotations

import json
from pathlib import Path

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/OVERLEAF_monthly_text_sentiment_overview_overleaf_figure_may22"
)
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
DATA_GLOB = (
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/data_/*.parquet"
)

SENTIMENT_ORDER = ["negative", "neutral", "positive"]
SENTIMENT_COLORS = {
    "negative": "#c0392b",
    "neutral": "#7f8c8d",
    "positive": "#2980b9",
}

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def build_monthly_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    monthly = con.execute(
        f"""
        WITH base AS (
            SELECT
                date_trunc('month', CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE)) AS month,
                "record.text.clean.sentimentlabel" AS sentiment
            FROM read_parquet('{DATA_GLOB}')
            WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
              AND "record.text.clean.sentimentlabel" IS NOT NULL
        )
        SELECT month, sentiment, count(*) AS post_count
        FROM base
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    monthly["month"] = pd.to_datetime(monthly["month"])
    monthly = monthly.sort_values(["month", "sentiment"]).reset_index(drop=True)
    monthly.to_csv(TABLE_DIR / "monthly_text_sentiment_counts.csv", index=False)

    totals = (
        monthly.groupby("month", as_index=False)["post_count"]
        .sum()
        .rename(columns={"post_count": "month_total"})
    )
    shares = monthly.merge(totals, on="month", how="left")
    shares["share_pct"] = shares["post_count"] / shares["month_total"] * 100.0
    shares.to_csv(TABLE_DIR / "monthly_text_sentiment_shares.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": ["total_rows", "min_month", "max_month", "distinct_sentiments"],
            "value": [
                int(monthly["post_count"].sum()),
                monthly["month"].min().strftime("%Y-%m-%d"),
                monthly["month"].max().strftime("%Y-%m-%d"),
                int(monthly["sentiment"].nunique()),
            ],
        }
    )
    summary.to_csv(TABLE_DIR / "monthly_text_sentiment_summary.csv", index=False)
    return monthly, shares, summary

def pivot_monthly(monthly: pd.DataFrame, value_col: str) -> pd.DataFrame:
    pivot = (
        monthly.pivot(index="month", columns="sentiment", values=value_col)
        .reindex(columns=SENTIMENT_ORDER)
        .fillna(0)
        .sort_index()
    )
    return pivot

def style_time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def add_segment_labels(ax: plt.Axes, x_vals, bottoms, heights, percent_mode: bool) -> None:
    for idx, _ in enumerate(x_vals):
        cumulative = 0.0
        total = sum(heights[s][idx] for s in SENTIMENT_ORDER)
        for sentiment in SENTIMENT_ORDER:
            height = heights[sentiment][idx]
            if height <= 0:
                cumulative += height
                continue
            pct = (height / total * 100.0) if total else 0.0
            if pct < 8:
                cumulative += height
                continue
            if percent_mode:
                label = f"{pct:.0f}%"
            else:
                label = f"{pct:.0f}%"
            y = cumulative + height / 2.0
            ax.text(
                x_vals[idx],
                y,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                fontweight="bold",
            )
            cumulative += height

def make_stacked_plot(
    pivot: pd.DataFrame,
    title: str,
    ylabel: str,
    out_name: str,
    percent_mode: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    x_vals = pivot.index.to_pydatetime()
    bottoms = [0.0] * len(pivot)
    heights = {sentiment: pivot[sentiment].tolist() for sentiment in SENTIMENT_ORDER}

    for sentiment in SENTIMENT_ORDER:
        ax.bar(
            x_vals,
            pivot[sentiment].values,
            bottom=bottoms,
            width=25,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            linewidth=0.5,
            label=sentiment.capitalize(),
        )
        bottoms = [b + h for b, h in zip(bottoms, heights[sentiment])]

    add_segment_labels(ax, x_vals, None, heights, percent_mode=percent_mode)
    ax.set_title(title, fontsize=18, pad=16, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xlabel("Month", fontsize=13)
    if percent_mode:
        ax.set_ylim(0, 100)
    style_time_axis(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=12, title="Sentiment", title_fontsize=12)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{out_name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    counts, shares, summary = build_monthly_tables()
    count_pivot = pivot_monthly(counts, "post_count")
    share_pivot = pivot_monthly(shares, "share_pct")
    make_stacked_plot(
        count_pivot,
        title="Monthly Bluesky Posts by Text Sentiment Label",
        ylabel="Posts",
        out_name="monthly_text_sentiment_stacked_counts",
        percent_mode=False,
    )
    make_stacked_plot(
        share_pivot,
        title="Monthly Bluesky Sentiment Composition by Text Sentiment Label",
        ylabel="Share of Monthly Posts (%)",
        out_name="monthly_text_sentiment_stacked_percent",
        percent_mode=True,
    )

if __name__ == "__main__":
    main()
