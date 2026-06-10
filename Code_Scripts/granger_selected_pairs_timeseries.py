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
OUTPUT_DIR = BASE_DIR / "selected_pairs_timeseries"

SELECTED_PAIRS = [
    (
        "Technology, Science, and Artificial Intelligence Development",
        "Technology, Platforms, and Digital Power",
    ),
    (
        "Education and Higher-Ed Transparency",
        "Education, Universities, and Intellectual Control",
    ),
    (
        "Infrastructure and Maritime Transportation Rules",
        "Economic Policy, Taxation, and Distributional Conflict",
    ),
    (
        "Foreign Policy and External Relations",
        "National Security, Military, and Geopolitical Conflict",
    ),
    (
        "National Security, Defense, and Strategic Resilience",
        "National Security, Military, and Geopolitical Conflict",
    ),
    (
        "Energy Development and Resource Expansion",
        "Public Lands, Environment, and Climate Governance",
    ),
    (
        "Health, Reproductive Care, and Medical Research",
        "Health, Public Health, and Bodily Politics",
    ),
    (
        "Executive Coordination and Administrative Task Forces",
        "Executive Power, Patronage, and Administrative Control",
    ),
]


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE_DIR / "eo19_to_bluesky_granger_input_daily.csv")
    data["day"] = pd.to_datetime(data["day"])

    selected_rows = []
    manifest_rows = []

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

    for i, (eo_theme, bs_theme) in enumerate(SELECTED_PAIRS, start=1):
        sub = data[
            (data["eo_theme"] == eo_theme)
            & (data["bluesky_theme"] == bs_theme)
        ].sort_values("day")
        selected_rows.append(sub)

        stem = f"{i:02d}_{slugify(eo_theme)}__to__{slugify(bs_theme)}"

        # Individual figure
        ind_fig, ind_ax = plt.subplots(figsize=(14, 6))
        ind_ax.plot(sub["day"], sub["eo_count"], color=eo_color, linewidth=2.0, label="EO count")
        ind_ax.set_ylabel("EO count", color=eo_color)
        ind_ax.tick_params(axis="y", labelcolor=eo_color)
        ind_ax.grid(True, axis="y", alpha=0.25)

        ind_ax2 = ind_ax.twinx()
        ind_ax2.plot(sub["day"], sub["bluesky_posts"], color=bs_color, linewidth=1.8, alpha=0.9, label="Bluesky posts")
        ind_ax2.set_ylabel("Bluesky posts", color=bs_color)
        ind_ax2.tick_params(axis="y", labelcolor=bs_color)

        ind_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ind_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ind_ax.tick_params(axis="x", rotation=30)
        ind_ax.set_title(f"{eo_theme}\nMapped Bluesky theme: {bs_theme}", loc="left", fontsize=16)
        handles = [
            plt.Line2D([0], [0], color=eo_color, lw=2, label="EO count"),
            plt.Line2D([0], [0], color=bs_color, lw=2, label="Bluesky posts"),
        ]
        ind_ax.legend(handles=handles, loc="upper left", frameon=True)
        ind_ax.set_xlabel("Date")
        ind_fig.tight_layout()
        ind_fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
        ind_fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(ind_fig)

        # Combined figure panel
        ax = axes[i - 1]
        ax.plot(sub["day"], sub["eo_count"], color=eo_color, linewidth=1.8)
        ax.set_ylabel("EO count", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(f"{i}. {eo_theme}\n{bs_theme}", fontsize=12, loc="left")

        ax2 = ax.twinx()
        ax2.plot(sub["day"], sub["bluesky_posts"], color=bs_color, linewidth=1.6, alpha=0.9)
        ax2.set_ylabel("Bluesky posts", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        manifest_rows.append(
            {
                "index": i,
                "eo_theme": eo_theme,
                "bluesky_theme": bs_theme,
                "png_file": f"{stem}.png",
                "pdf_file": f"{stem}.pdf",
            }
        )

    for ax in axes:
        ax.tick_params(axis="x", rotation=30)

    handles = [
        plt.Line2D([0], [0], color=eo_color, lw=2, label="EO count"),
        plt.Line2D([0], [0], color=bs_color, lw=2, label="Bluesky posts"),
    ]
    axes[0].legend(handles=handles, loc="upper left", frameon=True)
    fig.suptitle("Selected EO-to-Bluesky Relationship Time Series", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT_DIR / "selected_pairs_timeseries_grid.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "selected_pairs_timeseries_grid.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    selected_df = pd.concat(selected_rows, ignore_index=True)
    selected_df.to_csv(OUTPUT_DIR / "selected_pairs_timeseries_data.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "selected_pairs_manifest.csv", index=False)

    print(f"folder\t{OUTPUT_DIR}")
    print(f"grid\t{OUTPUT_DIR / 'selected_pairs_timeseries_grid.png'}")
    print(f"data\t{OUTPUT_DIR / 'selected_pairs_timeseries_data.csv'}")


if __name__ == "__main__":
    main()
