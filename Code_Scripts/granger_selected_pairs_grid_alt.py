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
from matplotlib.lines import Line2D


WORK_DIR = Path("local_data/03_Github_data/07_EO_effect_bluesky_2025_01_20_to_2026_02_01/eo19_to_bluesky_granger")
TABLE_DIR = WORK_DIR / "tables"
OUTPUT_DIR = WORK_DIR / "selected6_pairs_grid_set2"

SELECTED_PAIRS = [
    ("Domestic Investment and Market Competition", "Economic Policy, Taxation, and Distributional Conflict"),
    ("Environmental Stewardship and Beautification", "Public Lands, Environment, and Climate Governance"),
    ("Public Lands, Parks, and Timber Production", "Public Lands, Environment, and Climate Governance"),
    ("Workforce, Training, and Retirement Security", "Economic Policy, Taxation, and Distributional Conflict"),
    ("Health, Reproductive Care, and Medical Research", "Health, Public Health, and Bodily Politics"),
    ("National Security, Defense, and Strategic Resilience", "National Security, Military, and Geopolitical Conflict"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE_DIR / "eo19_to_bluesky_granger_input_daily.csv")
    data["day"] = pd.to_datetime(data["day"])

    selected_rows = []
    manifest = []
    for i, (eo_theme, bs_theme) in enumerate(SELECTED_PAIRS, start=1):
        sub = data[
            (data["eo_theme"] == eo_theme)
            & (data["bluesky_theme"] == bs_theme)
        ].sort_values("day").copy()
        sub["index"] = i
        selected_rows.append(sub)
        manifest.append({"index": i, "eo_theme": eo_theme, "bluesky_theme": bs_theme})

    selected_df = pd.concat(selected_rows, ignore_index=True)
    selected_df.to_csv(OUTPUT_DIR / "selected6_pairs_daily_data.csv", index=False)
    pd.DataFrame(manifest).to_csv(OUTPUT_DIR / "selected6_pairs_manifest.csv", index=False)

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

    fig, axes = plt.subplots(3, 2, figsize=(22, 16), sharex=True)
    axes = axes.flatten()

    for i, (eo_theme, bs_theme) in enumerate(SELECTED_PAIRS, start=1):
        ax = axes[i - 1]
        sub = selected_df[
            (selected_df["eo_theme"] == eo_theme)
            & (selected_df["bluesky_theme"] == bs_theme)
        ].sort_values("day")

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
        ax.set_title(f"{i}. {eo_theme}\n{bs_theme}", fontsize=12, loc="left")

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
    axes[0].legend(handles=handles, loc="upper left", frameon=True)

    fig.suptitle("Selected 6 EO-to-Bluesky Mappings: Daily Frequency Plots", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT_DIR / "selected6_pairs_daily_frequency_grid.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "selected6_pairs_daily_frequency_grid.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"folder\t{OUTPUT_DIR}")
    print(f"figure\t{OUTPUT_DIR / 'selected6_pairs_daily_frequency_grid.png'}")
    print(f"manifest\t{OUTPUT_DIR / 'selected6_pairs_manifest.csv'}")


if __name__ == "__main__":
    main()
