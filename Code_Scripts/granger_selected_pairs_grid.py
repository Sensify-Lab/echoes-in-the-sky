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


WORK_DIR = Path("local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs/bluesky_theme_to_eo_granger")
TABLE_DIR = WORK_DIR / "tables"
OUTPUT_DIR = WORK_DIR / "selected6_pairs_grid"

SELECTED_PAIRS = [
    ("Republican, MAGA, and Partisan Conflict", "Executive Coordination and Administrative Task Forces"),
    ("Technology, Platforms, and Digital Power", "Technology, Science, and Artificial Intelligence Development"),
    ("Education, Universities, and Intellectual Control", "Education and Higher-Ed Transparency"),
    ("National Security, Military, and Geopolitical Conflict", "National Security, Defense, and Strategic Resilience"),
    ("Health, Public Health, and Bodily Politics", "Health, Reproductive Care, and Medical Research"),
    ("Executive Power, Patronage, and Administrative Control", "Executive Coordination and Administrative Task Forces"),
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(TABLE_DIR / "bluesky_theme_to_eo_granger_input_daily.csv")
    data["day"] = pd.to_datetime(data["day"])

    selected_rows = []
    weekly_rows = []
    for i, (bs_theme, eo_theme) in enumerate(SELECTED_PAIRS, start=1):
        sub = data[
            (data["bluesky_theme"] == bs_theme)
            & (data["eo_theme"] == eo_theme)
        ].sort_values("day")
        sub = sub.copy()
        sub["index"] = i
        selected_rows.append(sub)

        weekly = (
            sub.set_index("day")[["eo_count", "bluesky_posts"]]
            .resample("W-MON")
            .sum()
            .reset_index()
            .rename(columns={"day": "week"})
        )
        weekly["bluesky_theme"] = bs_theme
        weekly["eo_theme"] = eo_theme
        weekly["index"] = i
        weekly_rows.append(weekly)

    selected_df = pd.concat(selected_rows, ignore_index=True)
    weekly_df = pd.concat(weekly_rows, ignore_index=True)
    selected_df.to_csv(OUTPUT_DIR / "selected6_pairs_daily_data.csv", index=False)
    weekly_df.to_csv(OUTPUT_DIR / "selected6_pairs_weekly_frequency_data.csv", index=False)

    manifest = pd.DataFrame(
        [{"index": i, "bluesky_theme": bs, "eo_theme": eo} for i, (bs, eo) in enumerate(SELECTED_PAIRS, start=1)]
    )
    manifest.to_csv(OUTPUT_DIR / "selected6_pairs_manifest.csv", index=False)

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

    for i, (bs_theme, eo_theme) in enumerate(SELECTED_PAIRS, start=1):
        ax = axes[i - 1]
        sub = selected_df[
            (selected_df["bluesky_theme"] == bs_theme)
            & (selected_df["eo_theme"] == eo_theme)
        ].sort_values("day")

        ind_fig, ind_ax = plt.subplots(figsize=(14, 6))
        ind_ax.plot(
            sub["day"],
            sub["eo_count"],
            color=eo_color,
            linewidth=2.8,
            marker="o",
            markersize=3.2,
            alpha=1.0,
        )
        ind_ax.set_ylabel("EO", color=eo_color)
        ind_ax.tick_params(axis="y", labelcolor=eo_color)
        ind_ax.grid(True, axis="y", alpha=0.25)
        ind_ax.set_title(f"{bs_theme}\n{eo_theme}", fontsize=16, loc="left")

        ind_ax2 = ind_ax.twinx()
        ind_ax2.plot(
            sub["day"],
            sub["bluesky_posts"],
            color=bs_color,
            linewidth=1.9,
            marker="o",
            markersize=3,
            alpha=0.9,
        )
        ind_ax2.set_ylabel("Bluesky", color=bs_color)
        ind_ax2.tick_params(axis="y", labelcolor=bs_color)
        ind_ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ind_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ind_ax.tick_params(axis="x", rotation=30)
        ind_handles = [
            Line2D([0], [0], color=eo_color, lw=2.8, marker="o", markersize=5, alpha=1.0, label="Daily EO count"),
            Line2D([0], [0], color=bs_color, lw=2, marker="o", markersize=5, label="Daily Bluesky posts"),
        ]
        ind_ax.legend(handles=ind_handles, loc="upper left", frameon=True)
        ind_fig.tight_layout()
        stem = f"{i:02d}_selected6_pair_daily"
        ind_fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
        ind_fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        plt.close(ind_fig)

        ax.plot(
            sub["day"],
            sub["eo_count"],
            color=eo_color,
            linewidth=2.6,
            marker="o",
            markersize=2.8,
            alpha=1.0,
        )
        ax.set_ylabel("EO", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(f"{i}. {bs_theme}\n{eo_theme}", fontsize=12, loc="left")

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

    fig.suptitle("Selected 6 Bluesky-to-EO Mappings: Daily Frequency Plots", fontsize=22, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(OUTPUT_DIR / "selected6_pairs_daily_frequency_grid.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTPUT_DIR / "selected6_pairs_daily_frequency_grid.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"folder\t{OUTPUT_DIR}")
    print(f"figure\t{OUTPUT_DIR / 'selected6_pairs_daily_frequency_grid.png'}")
    print(f"manifest\t{OUTPUT_DIR / 'selected6_pairs_manifest.csv'}")


if __name__ == "__main__":
    main()
