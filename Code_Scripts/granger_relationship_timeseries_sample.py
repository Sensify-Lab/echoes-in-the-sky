from __future__ import annotations

import os
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
FIG_DIR = BASE_DIR / "publication_figures"


def main() -> None:
    crosswalk = pd.read_csv(BASE_DIR / "eo19_to_bluesky_theme_crosswalk.csv").head(7).copy()
    data = pd.read_csv(TABLE_DIR / "eo19_to_bluesky_granger_input_daily.csv")
    data["day"] = pd.to_datetime(data["day"])

    selected = crosswalk.merge(
        data,
        on=["eo_theme", "bluesky_theme", "mapping_confidence", "mapping_notes"],
        how="left",
    )
    selected.to_csv(TABLE_DIR / "first7_relationship_timeseries_data.csv", index=False)

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

    fig, axes = plt.subplots(4, 2, figsize=(22, 18), sharex=True)
    axes = axes.flatten()

    eo_color = "#1f4e79"
    bs_color = "#c85108"

    for i, row in enumerate(crosswalk.itertuples(index=False)):
        ax = axes[i]
        sub = selected[
            (selected["eo_theme"] == row.eo_theme)
            & (selected["bluesky_theme"] == row.bluesky_theme)
        ].sort_values("day")

        ax.plot(
            sub["day"],
            sub["eo_count"],
            color=eo_color,
            linewidth=1.8,
            label="EO count",
        )
        ax.set_ylabel("EO count", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.set_title(
            f"{i + 1}. {row.eo_theme}\nvs {row.bluesky_theme}",
            fontsize=12,
            loc="left",
        )

        ax2 = ax.twinx()
        ax2.plot(
            sub["day"],
            sub["bluesky_posts"],
            color=bs_color,
            linewidth=1.6,
            alpha=0.9,
            label="Bluesky posts",
        )
        ax2.set_ylabel("Bluesky posts", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(True, axis="y", alpha=0.25)

        if i == 0:
            handles = [
                plt.Line2D([0], [0], color=eo_color, lw=2, label="EO count"),
                plt.Line2D([0], [0], color=bs_color, lw=2, label="Bluesky posts"),
            ]
            ax.legend(handles=handles, loc="upper left", frameon=True)

    axes[-1].axis("off")
    for ax in axes[:-1]:
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle(
        "First 7 EO-to-Bluesky Relationships: Daily Time Series",
        fontsize=22,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    stem = "first7_relationship_timeseries"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"data\t{TABLE_DIR / 'first7_relationship_timeseries_data.csv'}")
    print(f"figure\t{FIG_DIR / 'first7_relationship_timeseries.png'}")


if __name__ == "__main__":
    main()
