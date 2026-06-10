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
from matplotlib.lines import Line2D


WORK_DIR = Path("local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs/bluesky_theme_to_eo_granger")
TABLE_DIR = WORK_DIR / "tables"
OUTPUT_DIR = WORK_DIR / "all_pairs_daily_frequency"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_csv(WORK_DIR / "bluesky_to_eo_theme_crosswalk.csv")
    data = pd.read_csv(TABLE_DIR / "bluesky_theme_to_eo_granger_input_daily.csv")
    data["day"] = pd.to_datetime(data["day"])

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

    eo_color = "#7f0000"
    bs_color = "#1f77b4"
    selected_rows = []
    manifest_rows = []

    for i, row in enumerate(crosswalk.itertuples(index=False), start=1):
        sub = data[
            (data["bluesky_theme"] == row.bluesky_theme)
            & (data["eo_theme"] == row.eo_theme)
        ].sort_values("day").copy()
        sub["index"] = i
        selected_rows.append(sub)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            sub["day"],
            sub["eo_count"],
            color=eo_color,
            linewidth=2.8,
            marker="o",
            markersize=3.2,
            alpha=1.0,
        )
        ax.set_ylabel("EO", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(
            f"{i}. {row.bluesky_theme}\n{row.eo_theme}",
            fontsize=16,
            loc="left",
        )

        ax2 = ax.twinx()
        ax2.plot(
            sub["day"],
            sub["bluesky_posts"],
            color=bs_color,
            linewidth=1.9,
            marker="o",
            markersize=3,
            alpha=0.9,
        )
        ax2.set_ylabel("Bluesky", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        handles = [
            Line2D([0], [0], color=eo_color, lw=2.8, marker="o", markersize=5, alpha=1.0, label="Daily EO count"),
            Line2D([0], [0], color=bs_color, lw=2, marker="o", markersize=5, label="Daily Bluesky posts"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True)
        fig.tight_layout()

        stem = f"{i:02d}_{slugify(row.bluesky_theme)}__to__{slugify(row.eo_theme)}_daily"
        png_path = OUTPUT_DIR / f"{stem}.png"
        pdf_path = OUTPUT_DIR / f"{stem}.pdf"
        fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        manifest_rows.append(
            {
                "index": i,
                "bluesky_theme": row.bluesky_theme,
                "eo_theme": row.eo_theme,
                "mapping_confidence": row.mapping_confidence,
                "mapping_notes": row.mapping_notes,
                "png_file": png_path.name,
                "pdf_file": pdf_path.name,
            }
        )

    daily_df = pd.concat(selected_rows, ignore_index=True)
    daily_df.to_csv(OUTPUT_DIR / "all_pairs_daily_frequency_data.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "all_pairs_daily_frequency_manifest.csv", index=False)

    print(f"folder\t{OUTPUT_DIR}")
    print(f"manifest\t{OUTPUT_DIR / 'all_pairs_daily_frequency_manifest.csv'}")
    print(f"data\t{OUTPUT_DIR / 'all_pairs_daily_frequency_data.csv'}")


if __name__ == "__main__":
    main()
