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


BASE_DIR = Path("local_data/03_Github_data/07_EO_effect_bluesky_2025_01_20_to_2026_02_01")
WORK_DIR = BASE_DIR / "bluesky_theme_to_eo_granger"
TABLE_DIR = BASE_DIR / "tables"
OUTPUT_DIR = WORK_DIR / "eo_bluesky_frequency_plots"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_csv(BASE_DIR / "eo_bluesky_theme_crosswalk.csv")
    eo_daily = pd.read_csv(TABLE_DIR / "eo_theme_daily.csv")
    bs_daily = pd.read_csv(TABLE_DIR / "bluesky_theme_daily.csv")
    eo_daily["day"] = pd.to_datetime(eo_daily["day"])
    bs_daily["day"] = pd.to_datetime(bs_daily["day"])

    daily_rows = []
    weekly_rows = []

    for row in crosswalk.itertuples(index=False):
        eo_sub = (
            eo_daily[eo_daily["eo_theme"] == row.eo_theme]
            .groupby("day", as_index=False)["eo_count"]
            .sum()
        )
        bs_sub = (
            bs_daily[bs_daily["bluesky_theme"] == row.bluesky_theme]
            .groupby("day", as_index=False)["bluesky_posts"]
            .sum()
        )

        merged = pd.merge(eo_sub, bs_sub, on="day", how="outer").fillna(0).sort_values("day")
        merged["eo_theme"] = row.eo_theme
        merged["bluesky_theme"] = row.bluesky_theme
        merged["mapping_confidence"] = row.mapping_confidence
        daily_rows.append(merged)

        weekly = (
            merged.set_index("day")[["eo_count", "bluesky_posts"]]
            .resample("W-MON")
            .sum()
            .reset_index()
            .rename(columns={"day": "week"})
        )
        weekly["eo_theme"] = row.eo_theme
        weekly["bluesky_theme"] = row.bluesky_theme
        weekly["mapping_confidence"] = row.mapping_confidence
        weekly_rows.append(weekly)

    daily_df = pd.concat(daily_rows, ignore_index=True)
    weekly_df = pd.concat(weekly_rows, ignore_index=True)
    daily_df.to_csv(OUTPUT_DIR / "eo_bluesky_daily_frequency_data.csv", index=False)
    weekly_df.to_csv(OUTPUT_DIR / "eo_bluesky_weekly_frequency_data.csv", index=False)
    crosswalk.to_csv(OUTPUT_DIR / "eo_bluesky_theme_crosswalk_used.csv", index=False)

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
    manifest_rows = []

    for i, row in enumerate(crosswalk.itertuples(index=False), start=1):
        sub = weekly_df[
            (weekly_df["eo_theme"] == row.eo_theme)
            & (weekly_df["bluesky_theme"] == row.bluesky_theme)
        ].sort_values("week")

        stem = f"{i:02d}_{slugify(row.eo_theme)}__to__{slugify(row.bluesky_theme)}_weekly_frequency"
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar(
            sub["week"],
            sub["eo_count"],
            width=5,
            color=eo_color,
            alpha=0.75,
            label="Weekly EO count",
        )
        ax.set_ylabel("Weekly EO count", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(
            sub["week"],
            sub["bluesky_posts"],
            color=bs_color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            alpha=0.9,
            label="Weekly Bluesky posts",
        )
        ax2.set_ylabel("Weekly Bluesky posts", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        ax.set_title(f"{row.eo_theme}\nMapped Bluesky theme: {row.bluesky_theme}", loc="left", fontsize=16)
        handles = [
            plt.Line2D([0], [0], color=eo_color, lw=6, alpha=0.75, label="Weekly EO count"),
            plt.Line2D([0], [0], color=bs_color, lw=2, marker="o", markersize=5, label="Weekly Bluesky posts"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True)
        ax.set_xlabel("Week")
        fig.tight_layout()

        png_path = OUTPUT_DIR / f"{stem}.png"
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        manifest_rows.append(
            {
                "index": i,
                "eo_theme": row.eo_theme,
                "bluesky_theme": row.bluesky_theme,
                "mapping_confidence": row.mapping_confidence,
                "png_file": png_path.name,
                "pdf_file": pdf_path.name,
            }
        )

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUTPUT_DIR / "eo_bluesky_frequency_manifest.csv", index=False)
    print(f"folder\t{OUTPUT_DIR}")
    print(f"manifest\t{OUTPUT_DIR / 'eo_bluesky_frequency_manifest.csv'}")
    print(f"weekly_data\t{OUTPUT_DIR / 'eo_bluesky_weekly_frequency_data.csv'}")


if __name__ == "__main__":
    main()
