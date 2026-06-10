from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import matplotlib

matplotlib.use("Agg")

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter

BASE_DIR = Path("local_data/03_Github_data/04_final_merged_10M")
ANALYSIS_DIR = BASE_DIR / "analysis_outputs"
PUB_DIR = ANALYSIS_DIR / "publication_figures"
SELECTED_DIR = ANALYSIS_DIR / "selected_figures"
PARQUET_GLOB = str(BASE_DIR / "*.parquet")

SELECTED_FIGURES = [
    "dataset_overview_kpi",
    "theme_distribution_barh",
    "theme_share_barh",
    "monthly_volume_area",
    "monthly_volume_heatmap",
    "top8_theme_monthly_lines",
    "top8_theme_stacked_area",
    "theme_year_heatmap",
    "top_authors_theme_mix_stacked",
]

def fmt_int(value: float, _pos: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.0f}K"
    return f"{value:.0f}"

def ensure_dirs() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

def copy_selected_figures() -> list[str]:
    copied = []
    for stem in SELECTED_FIGURES:
        for ext in ("png", "pdf"):
            src = PUB_DIR / f"{stem}.{ext}"
            dst = SELECTED_DIR / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, dst)
                copied.append(dst.name)
    return copied

def generate_theme_scale_plot() -> list[str]:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0F172A",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )

    df = pd.read_csv(ANALYSIS_DIR / "theme_author_summary.csv").copy()
    df["posts_per_author"] = df["rows"] / df["distinct_authors"]
    df["theme_label"] = df["theme_name"].str.replace(", and ", ",\n", regex=False)

    fig, ax = plt.subplots(figsize=(12.5, 8.5))
    scatter = ax.scatter(
        df["distinct_authors"],
        df["rows"],
        s=np.clip(df["avg_likes"] * 26, 110, 850),
        c=df["posts_per_author"],
        cmap="YlOrRd",
        alpha=0.92,
        edgecolor="white",
        linewidth=1.3,
    )

    x_pad = df["distinct_authors"].max() * 0.03
    y_pad = df["rows"].max() * 0.03
    for _, row in df.iterrows():
        ax.text(
            row["distinct_authors"] + x_pad * 0.15,
            row["rows"] + y_pad * 0.04,
            row["theme_label"],
            fontsize=9.2,
            ha="left",
            va="bottom",
        )

    ax.set_title("Theme Scale: Authors vs Posts", pad=16, fontweight="bold")
    ax.set_xlabel("Distinct Authors")
    ax.set_ylabel("Posts")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.grid(True, axis="both", color="#E2E8F0", linewidth=0.8)

    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Posts per Author", rotation=90)

    size_levels = [120, 350, 700]
    size_labels = ["~5 avg likes", "~15 avg likes", "~30+ avg likes"]
    handles = [
        ax.scatter([], [], s=s, color="#FB923C", alpha=0.7, edgecolor="white", linewidth=1.0)
        for s in size_levels
    ]
    ax.legend(handles, size_labels, title="Marker Size", loc="lower right")

    note = (
        "X-axis: unique authors in theme | Y-axis: post volume | "
        "Color: posts per author | Size: average likes"
    )
    fig.text(0.01, 0.01, note, fontsize=9, color="#475569")

    stems = []
    for ext in ("png", "pdf"):
        out = SELECTED_DIR / f"theme_scale_authors_vs_posts_regenerated.{ext}"
        fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
        stems.append(out.name)
    plt.close(fig)
    return stems

def save_fig(fig: plt.Figure, stem: str) -> list[str]:
    saved = []
    for ext in ("png", "pdf"):
        out = SELECTED_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=320, bbox_inches="tight", facecolor="white")
        saved.append(out.name)
    plt.close(fig)
    return saved

def query_theme_time_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    month_df = con.execute(
        f"""
        SELECT
          date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS period,
          "Theme_Name" AS theme_name,
          count(*) AS rows
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """
    ).df()
    quarter_df = con.execute(
        f"""
        SELECT
          date_trunc('quarter', try_cast("record.created_at" AS TIMESTAMPTZ)) AS period,
          "Theme_Name" AS theme_name,
          count(*) AS rows
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """
    ).df()
    day_df = con.execute(
        f"""
        SELECT
          CAST(date_trunc('day', try_cast("record.created_at" AS TIMESTAMPTZ)) AS DATE) AS period,
          "Theme_Name" AS theme_name,
          count(*) AS rows
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
          AND try_cast("record.created_at" AS TIMESTAMPTZ) >= TIMESTAMPTZ '2024-11-01'
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """
    ).df()
    weekday_df = con.execute(
        f"""
        SELECT
          strftime(try_cast("record.created_at" AS TIMESTAMPTZ), '%w') AS weekday_num,
          strftime(try_cast("record.created_at" AS TIMESTAMPTZ), '%A') AS weekday_name,
          "Theme_Name" AS theme_name,
          count(*) AS rows
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
        GROUP BY 1, 2, 3
        ORDER BY 1, 4 DESC
        """
    ).df()
    return month_df, quarter_df, day_df, weekday_df

def ordered_pivot(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    pivot = df.pivot(index="theme_name", columns=period_label, values="rows").fillna(0)
    order = df.groupby("theme_name")["rows"].sum().sort_values(ascending=False).index
    return pivot.loc[order]

def generate_theme_time_figures() -> list[str]:
    sns.set_theme(style="whitegrid", context="talk")
    saved: list[str] = []
    month_df, quarter_df, day_df, weekday_df = query_theme_time_tables()

    month_df["period"] = pd.to_datetime(month_df["period"], utc=True).dt.tz_convert(None)
    month_df["period_label"] = month_df["period"].dt.strftime("%Y-%m")
    month_pivot = ordered_pivot(month_df, "period_label")
    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(month_pivot, cmap="YlOrBr", linewidths=0.15, linecolor="white", cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    ax.set_title("Theme Distribution by Month", pad=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    saved += save_fig(fig, "theme_distribution_by_month_heatmap")

    quarter_df["period"] = pd.to_datetime(quarter_df["period"], utc=True).dt.tz_convert(None)
    quarter_df["period_label"] = quarter_df["period"].dt.year.astype(str) + "-Q" + quarter_df["period"].dt.quarter.astype(str)
    quarter_pivot = ordered_pivot(quarter_df, "period_label")
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(quarter_pivot, cmap="crest", linewidths=0.2, linecolor="white", annot=True, fmt=".0f", cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    ax.set_title("Theme Distribution by Quarter", pad=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    saved += save_fig(fig, "theme_distribution_by_quarter_heatmap")

    day_df["period"] = pd.to_datetime(day_df["period"])
    recent_day_pivot = ordered_pivot(day_df.assign(period_label=day_df["period"].dt.strftime("%Y-%m-%d")), "period_label")
    fig, ax = plt.subplots(figsize=(24, 8))
    sns.heatmap(recent_day_pivot, cmap="rocket_r", linewidths=0.0, cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    tick_positions = np.arange(0, recent_day_pivot.shape[1], 14) + 0.5
    tick_labels = [recent_day_pivot.columns[i] for i in range(0, recent_day_pivot.shape[1], 14)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax.set_title("Theme Distribution by Day", pad=14, fontweight="bold")
    ax.set_xlabel("Daily bins from 2024-11-01 onward")
    ax.set_ylabel("")
    saved += save_fig(fig, "theme_distribution_by_day_heatmap")

    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekday_df["weekday_name"] = pd.Categorical(weekday_df["weekday_name"], categories=weekday_order, ordered=True)
    weekday_pivot = weekday_df.pivot(index="theme_name", columns="weekday_name", values="rows").fillna(0)
    weekday_pivot = weekday_pivot[weekday_order]
    weekday_pivot = weekday_pivot.loc[weekday_df.groupby("theme_name")["rows"].sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        weekday_pivot,
        cmap="PuRd",
        linewidths=0.4,
        linecolor="white",
        annot=True,
        fmt=".0f",
        cbar_kws={"format": FuncFormatter(fmt_int)},
        ax=ax,
    )
    ax.set_title("Theme Distribution by Weekday", pad=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    saved += save_fig(fig, "theme_distribution_by_weekday_heatmap")

    return saved

def save_manifest(files: list[str]) -> str:
    manifest = SELECTED_DIR / "selected_figures_manifest.txt"
    manifest.write_text("\n".join(sorted(files)) + "\n", encoding="utf-8")
    return manifest.name

def main() -> None:
    ensure_dirs()
    copied = copy_selected_figures()
    time_figures = generate_theme_time_figures()
    regenerated = generate_theme_scale_plot()
    manifest = save_manifest(copied + time_figures + regenerated)
    print(
        f"Saved {len(copied)} copied figures, {len(time_figures)} time-distribution files, "
        f"{len(regenerated)} regenerated files, to {SELECTED_DIR}"
    )
    print(f"Manifest: {manifest}")

if __name__ == "__main__":
    main()
