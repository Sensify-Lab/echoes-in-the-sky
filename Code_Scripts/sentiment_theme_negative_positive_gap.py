from __future__ import annotations

from pathlib import Path
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SOURCE_TABLE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_sentiment_changes/tables"
)
BASE_DIR = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/"
    "OVERLEAF_monthly_negative_positive_gap_per_theme_may22"
)
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_gap_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    shares = pd.read_csv(
        SOURCE_TABLE_DIR / "theme_sentiment_monthly_shares.csv",
        parse_dates=["month"],
    )
    shares = shares.loc[shares["month"] >= "2023-06-01"].copy()

    wide = (
        shares.pivot_table(
            index=["month", "theme"],
            columns="sentiment",
            values="share_pct",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
    )

    for col in ["negative", "neutral", "positive"]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["negative_minus_positive"] = wide["negative"] - wide["positive"]
    wide = wide.sort_values(["theme", "month"]).reset_index(drop=True)

    summary = (
        wide.groupby("theme", as_index=False)
        .agg(
            n_months=("month", "nunique"),
            series_start=("month", "min"),
            series_end=("month", "max"),
            avg_gap=("negative_minus_positive", "mean"),
            min_gap=("negative_minus_positive", "min"),
            max_gap=("negative_minus_positive", "max"),
        )
        .sort_values("theme")
    )

    shares.to_csv(TABLE_DIR / "theme_monthly_text_sentiment_shares.csv", index=False)
    wide.to_csv(TABLE_DIR / "theme_monthly_negative_minus_positive_gap.csv", index=False)
    summary.to_csv(TABLE_DIR / "theme_monthly_negative_minus_positive_gap_summary.csv", index=False)
    return wide, summary

def style_axis(ax) -> None:
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

def plot_gap(theme_df: pd.DataFrame, theme: str, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 3.8))

    ax.plot(
        theme_df["month"],
        theme_df["negative_minus_positive"],
        color="#8e2a1e",
        linewidth=2.2,
    )
    ax.axhline(0, color="#7f8c8d", linewidth=1.0, linestyle="--")

    style_axis(ax)
    ax.set_ylabel(
        "Negative - Positive (%)",
        fontsize=16,
        fontweight="bold",
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

def plot_gap_all_themes(gap_df: pd.DataFrame, fig_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))

    for theme in sorted(gap_df["theme"].dropna().unique()):
        theme_df = gap_df.loc[gap_df["theme"] == theme].copy()
        ax.plot(
            theme_df["month"],
            theme_df["negative_minus_positive"],
            linewidth=1.4,
            alpha=0.9,
            label=theme,
        )

    ax.axhline(0, color="#7f8c8d", linewidth=1.0, linestyle="--")
    style_axis(ax)
    ax.set_ylabel(
        "Negative - Positive (%)",
        fontsize=16,
        fontweight="bold",
    )
    ax.tick_params(axis="both", labelsize=14)
    ax.legend(
        frameon=False,
        fontsize=9.5,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handlelength=1.8,
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

def build_figures(gap_df: pd.DataFrame) -> None:
    for theme in sorted(gap_df["theme"].dropna().unique()):
        theme_df = gap_df.loc[gap_df["theme"] == theme].copy()
        plot_gap(
            theme_df,
            theme,
            FIG_DIR / f"{slugify(theme)}_monthly_negative_minus_positive_gap.png",
        )
    plot_gap_all_themes(
        gap_df,
        FIG_DIR / "all_themes_monthly_negative_minus_positive_gap.png",
    )

def main() -> None:
    ensure_dirs()
    gap_df, _summary = load_gap_table()
    build_figures(gap_df)

if __name__ == "__main__":
    main()
