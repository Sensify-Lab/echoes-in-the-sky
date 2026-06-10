from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path("local_data/03_Github_data/07_EO_effect_bluesky_2025_01_20_to_2026_02_01/eo19_to_bluesky_granger")
SOURCE_DIR = BASE_DIR / "selected_pairs_timeseries"
OUTPUT_DIR = BASE_DIR / "selected_pairs_frequency_plots"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(SOURCE_DIR / "selected_pairs_manifest.csv")
    data = pd.read_csv(SOURCE_DIR / "selected_pairs_timeseries_data.csv")
    data["day"] = pd.to_datetime(data["day"])

    weekly_rows = []
    for row in manifest.itertuples(index=False):
        sub = data[
            (data["eo_theme"] == row.eo_theme)
            & (data["bluesky_theme"] == row.bluesky_theme)
        ].sort_values("day")
        weekly = (
            sub.set_index("day")[["eo_count", "bluesky_posts"]]
            .resample("W-MON")
            .sum()
            .reset_index()
            .rename(columns={"day": "week"})
        )
        weekly["eo_theme"] = row.eo_theme
        weekly["bluesky_theme"] = row.bluesky_theme
        weekly["index"] = row.index
        weekly_rows.append(weekly)

    weekly_df = pd.concat(weekly_rows, ignore_index=True)
    weekly_df.to_csv(OUTPUT_DIR / "selected_pairs_weekly_frequency_data.csv", index=False)
    manifest.to_csv(OUTPUT_DIR / "selected_pairs_manifest.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    eo_color = "#1f4e79"
    bs_color = "#c85108"

    fig, axes = plt.subplots(4, 2, figsize=(22, 20), sharex=True)
    axes = axes.flatten()

    for i, row in enumerate(manifest.itertuples(index=False), start=1):
        sub = weekly_df[
            (weekly_df["eo_theme"] == row.eo_theme)
            & (weekly_df["bluesky_theme"] == row.bluesky_theme)
        ].sort_values("week")

        stem = f"{i:02d}_{slugify(row.eo_theme)}__to__{slugify(row.bluesky_theme)}_weekly_frequency"

        ind_fig, ind_ax = plt.subplots(figsize=(14, 6))
        ind_ax.bar(
            sub["week"],
            sub["eo_count"],
            width=5,
            color=eo_color,
            alpha=0.75,
            label="Weekly EO count",
        )
        ind_ax.set_ylabel("Weekly EO count", color=eo_color)
        ind_ax.tick_params(axis="y", labelcolor=eo_color)
        ind_ax.grid(True, axis="y", alpha=0.25)

        ind_ax2 = ind_ax.twinx()
        ind_ax2.plot(
            sub["week"],
            sub["bluesky_posts"],
            color=bs_color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            alpha=0.9,
            label="Weekly Bluesky posts",
        )
        ind_ax2.set_ylabel("Weekly Bluesky posts", color=bs_color)
        ind_ax2.tick_params(axis="y", labelcolor=bs_color)

        ind_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ind_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ind_ax.tick_params(axis="x", rotation=30)
        ind_ax.set_title(f"{row.eo_theme}\nMapped Bluesky theme: {row.bluesky_theme}", loc="left", fontsize=16)
        handles = [
            plt.Line2D([0], [0], color=eo_color, lw=6, alpha=0.75, label="Weekly EO count"),
            plt.Line2D([0], [0], color=bs_color, lw=2, marker="o", markersize=5, label="Weekly Bluesky posts"),
        ]
        ind_ax.legend(handles=handles, loc="upper left", frameon=True)
        ind_ax.set_xlabel("Week")
        ind_fig.tight_layout()
        ind_fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
        ind_fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(ind_fig)

        ax = axes[i - 1]
        ax.bar(sub["week"], sub["eo_count"], width=5, color=eo_color, alpha=0.75)
        ax.set_ylabel("EO", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(f"{i}. {row.eo_theme}\n{row.bluesky_theme}", fontsize=12, loc="left")

        ax2 = ax.twinx()
        ax2.plot(sub["week"], sub["bluesky_posts"], color=bs_color, linewidth=1.8, marker="o", markersize=2.8, alpha=0.9)
        ax2.set_ylabel("Bluesky", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    handles = [
        plt.Line2D([0], [0], color=eo_color, lw=6, alpha=0.75, label="Weekly EO count"),
        plt.Line2D([0], [0], color=bs_color, lw=2, marker="o", markersize=5, label="Weekly Bluesky posts"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True)
    fig.suptitle("Selected EO-to-Bluesky Weekly Frequency Plots", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT_DIR / "selected_pairs_weekly_frequency_grid.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "selected_pairs_weekly_frequency_grid.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"folder\t{OUTPUT_DIR}")
    print(f"grid\t{OUTPUT_DIR / 'selected_pairs_weekly_frequency_grid.png'}")
    print(f"data\t{OUTPUT_DIR / 'selected_pairs_weekly_frequency_data.csv'}")


if __name__ == "__main__":
    main()
