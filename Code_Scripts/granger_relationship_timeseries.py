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
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "all_relationship_timeseries"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    crosswalk = pd.read_csv(BASE_DIR / "eo19_to_bluesky_theme_crosswalk.csv")
    data = pd.read_csv(TABLE_DIR / "eo19_to_bluesky_granger_input_daily.csv")
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

    manifest_rows = []
    eo_color = "#1f4e79"
    bs_color = "#c85108"

    for i, row in enumerate(crosswalk.itertuples(index=False), start=1):
        sub = data[
            (data["eo_theme"] == row.eo_theme)
            & (data["bluesky_theme"] == row.bluesky_theme)
        ].sort_values("day")

        stem = f"{i:02d}_{slugify(row.eo_theme)}__to__{slugify(row.bluesky_theme)}"

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(
            sub["day"],
            sub["eo_count"],
            color=eo_color,
            linewidth=2.0,
            label="EO count",
        )
        ax.set_ylabel("EO count", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)

        ax2 = ax.twinx()
        ax2.plot(
            sub["day"],
            sub["bluesky_posts"],
            color=bs_color,
            linewidth=1.8,
            alpha=0.9,
            label="Bluesky posts",
        )
        ax2.set_ylabel("Bluesky posts", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)

        handles = [
            plt.Line2D([0], [0], color=eo_color, lw=2, label="EO count"),
            plt.Line2D([0], [0], color=bs_color, lw=2, label="Bluesky posts"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=True)

        ax.set_title(
            f"{i}. {row.eo_theme}\nMapped Bluesky theme: {row.bluesky_theme}",
            loc="left",
            fontsize=16,
        )
        ax.set_xlabel("Date")

        fig.tight_layout()
        png_path = FIG_DIR / f"{stem}.png"
        pdf_path = FIG_DIR / f"{stem}.pdf"
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
    manifest.to_csv(FIG_DIR / "relationship_timeseries_manifest.csv", index=False)
    print(f"manifest\t{FIG_DIR / 'relationship_timeseries_manifest.csv'}")
    print(f"figure_dir\t{FIG_DIR}")


if __name__ == "__main__":
    main()
