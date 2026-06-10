from __future__ import annotations

from math import ceil
from pathlib import Path
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

SOURCE_DIR = Path(
    "local_data/03_Github_data/11_Granger_causality_heatmaps/"
    "full_matrix_strict_stationary_only_no_na_sigcount"
)
INPUT_CSV = SOURCE_DIR / "tables" / "full_matrix_granger_input_daily_strict.csv"

BASE_DIR = SOURCE_DIR / "corrected_pair_timeseries_from_final_figure7"
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
PAIR_DIR = FIG_DIR / "per_pair_two_tier"
BS_OVERVIEW_DIR = FIG_DIR / "per_bluesky_theme_large_panels"

WINDOW_START = pd.Timestamp("2025-01-20")
WINDOW_END = pd.Timestamp("2026-02-01")

PAIR_FIGSIZE = (12, 5.4)
PAIR_DPI = 320
PANEL_DPI = 320
PANEL_NCOLS = 5
CELL_WIDTH = 3.0
CELL_HEIGHT = 2.1

BLUESKY_COLOR = "#1f5fa8"
EO_COLOR = "#8e2a1e"
GRID_COLOR = "#d9d9d9"
TITLE_COLOR = "#222222"

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    PAIR_DIR.mkdir(parents=True, exist_ok=True)
    BS_OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)

def load_panel() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["day"])
    df = df.sort_values(["bluesky_theme", "eo_theme", "day"]).reset_index(drop=True)
    df.to_csv(TABLE_DIR / "full_matrix_granger_input_daily_strict_copy.csv", index=False)
    return df

def style_ts_axis(ax, show_xlabels: bool) -> None:
    ax.set_xlim(WINDOW_START, WINDOW_END)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, alpha=0.7)
    ax.tick_params(axis="both", labelsize=8)
    if show_xlabels:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    else:
        ax.tick_params(axis="x", labelbottom=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_pair_figure(pair_df: pd.DataFrame, out_path: Path) -> None:
    meta = pair_df.iloc[0]
    fig = plt.figure(figsize=PAIR_FIGSIZE)
    gs = GridSpec(2, 1, figure=fig, hspace=0.12)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_top)

    ax_top.plot(pair_df["day"], pair_df["bluesky_posts_stationary"], color=BLUESKY_COLOR, linewidth=1.4)
    ax_bottom.plot(pair_df["day"], pair_df["eo_count_stationary"], color=EO_COLOR, linewidth=1.4)

    ax_top.axhline(0, color="#888888", linestyle="--", linewidth=0.8)
    ax_bottom.axhline(0, color="#888888", linestyle="--", linewidth=0.8)

    style_ts_axis(ax_top, show_xlabels=False)
    style_ts_axis(ax_bottom, show_xlabels=True)

    ax_top.set_ylabel("Bluesky\ncorrected", fontsize=10, fontweight="bold")
    ax_bottom.set_ylabel("EO\ncorrected", fontsize=10, fontweight="bold")

    title = (
        f"{meta['bluesky_theme']}\n"
        f"vs {meta['eo_theme']}\n"
        f"BS={meta['bluesky_posts_transform_used']} | EO={meta['eo_count_transform_used']} | "
        f"strict_ready={bool(meta['pair_ready_strict'])}"
    )
    fig.suptitle(title, fontsize=12, fontweight="bold", color=TITLE_COLOR, y=0.98)
    fig.savefig(out_path, dpi=PAIR_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_bluesky_theme_overview(bluesky_theme: str, theme_df: pd.DataFrame, out_path: Path) -> None:
    eo_themes = sorted(theme_df["eo_theme"].dropna().unique())
    n_pairs = len(eo_themes)
    ncols = PANEL_NCOLS
    nrows = ceil(n_pairs / ncols)

    fig = plt.figure(figsize=(ncols * CELL_WIDTH, nrows * CELL_HEIGHT * 2.0))
    outer = fig.add_gridspec(nrows, ncols, wspace=0.28, hspace=0.34)

    for idx, eo_theme in enumerate(eo_themes):
        row = idx // ncols
        col = idx % ncols
        cell = outer[row, col].subgridspec(2, 1, hspace=0.08)
        ax_top = fig.add_subplot(cell[0, 0])
        ax_bottom = fig.add_subplot(cell[1, 0], sharex=ax_top)

        pair_df = theme_df[theme_df["eo_theme"] == eo_theme].copy()
        meta = pair_df.iloc[0]

        ax_top.plot(pair_df["day"], pair_df["bluesky_posts_stationary"], color=BLUESKY_COLOR, linewidth=0.9)
        ax_bottom.plot(pair_df["day"], pair_df["eo_count_stationary"], color=EO_COLOR, linewidth=0.9)
        ax_top.axhline(0, color="#888888", linestyle="--", linewidth=0.5)
        ax_bottom.axhline(0, color="#888888", linestyle="--", linewidth=0.5)

        style_ts_axis(ax_top, show_xlabels=False)
        style_ts_axis(ax_bottom, show_xlabels=(row == nrows - 1))
        ax_top.tick_params(axis="both", labelsize=6)
        ax_bottom.tick_params(axis="both", labelsize=6)
        if row != nrows - 1:
            ax_bottom.tick_params(axis="x", labelbottom=False)

        ax_top.set_title(
            f"{eo_theme}\nBS={meta['bluesky_posts_transform_used']} | EO={meta['eo_count_transform_used']}",
            fontsize=7.3,
            fontweight="bold",
            pad=2,
        )
        ax_top.set_ylabel("BS", fontsize=6.5, fontweight="bold")
        ax_bottom.set_ylabel("EO", fontsize=6.5, fontweight="bold")

        if not bool(meta["pair_ready_strict"]):
            ax_top.text(
                0.98, 0.88, "not strict-ready",
                transform=ax_top.transAxes,
                ha="right", va="top",
                fontsize=6.2, color="#7f8c8d", fontstyle="italic"
            )

    total_cells = nrows * ncols
    for idx in range(n_pairs, total_cells):
        row = idx // ncols
        col = idx % ncols
        ax_blank = fig.add_subplot(outer[row, col])
        ax_blank.axis("off")

    fig.suptitle(
        f"{bluesky_theme}: corrected pair time series against all EO themes",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.savefig(out_path, dpi=PANEL_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

def build_manifest(panel: pd.DataFrame) -> pd.DataFrame:
    manifest = (
        panel.groupby(["bluesky_theme", "eo_theme"], as_index=False)
        .agg(
            series_start=("day", "min"),
            series_end=("day", "max"),
            eo_transform_used=("eo_count_transform_used", "first"),
            bluesky_transform_used=("bluesky_posts_transform_used", "first"),
            pair_ready_strict=("pair_ready_strict", "first"),
            n_rows=("day", "size"),
        )
        .sort_values(["bluesky_theme", "eo_theme"])
    )
    manifest.to_csv(TABLE_DIR / "corrected_pair_manifest.csv", index=False)
    return manifest

def main() -> None:
    ensure_dirs()
    panel = load_panel()
    manifest = build_manifest(panel)

    for (bluesky_theme, eo_theme), pair_df in panel.groupby(["bluesky_theme", "eo_theme"], sort=True):
        pair_stem = f"{slugify(bluesky_theme)}__{slugify(eo_theme)}_corrected_pair_timeseries"
        plot_pair_figure(pair_df, PAIR_DIR / f"{pair_stem}.png")

    for bluesky_theme, theme_df in panel.groupby("bluesky_theme", sort=True):
        panel_stem = f"{slugify(bluesky_theme)}_all_eo_pairs_corrected_large_panel"
        plot_bluesky_theme_overview(bluesky_theme, theme_df, BS_OVERVIEW_DIR / f"{panel_stem}.png")

if __name__ == "__main__":
    main()
