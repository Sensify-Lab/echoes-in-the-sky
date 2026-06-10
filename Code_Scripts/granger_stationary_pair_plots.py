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


BASE_DIR = Path("local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs")
STATIONARY_DIR = BASE_DIR / "stationary_granger_analysis_corrected_May20"
TABLE_DIR = STATIONARY_DIR / "tables"
OUT_DIR = STATIONARY_DIR / "per_pair_stationary_series"

INPUT_CSV = BASE_DIR / "stationarity_analysis" / "tables" / "granger_stationary_input_recommended.csv"
TESTED_B2E = TABLE_DIR / "bluesky_to_eo_tested_pairs_stationary.csv"
SKIPPED = TABLE_DIR / "granger_skipped_pairs_stationary.csv"


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ensure_dirs()
    df = pd.read_csv(INPUT_CSV)
    df["day"] = pd.to_datetime(df["day"])

    tested = pd.read_csv(TESTED_B2E)
    tested_pairs = set(tested["pair_id"].unique())

    skipped = pd.read_csv(SKIPPED)
    skipped_pairs = set(skipped["pair_id"].unique()) if not skipped.empty else set()

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

    eo_color = "#c0392b"
    bs_color = "#1f77b4"
    manifest_rows = []
    out_rows = []

    pair_keys = (
        df[["bluesky_theme", "eo_theme"]]
        .drop_duplicates()
        .sort_values(["bluesky_theme", "eo_theme"])
        .itertuples(index=False)
    )

    for i, key in enumerate(pair_keys, start=1):
        bluesky_theme = key.bluesky_theme
        eo_theme = key.eo_theme
        pair_id = f"{bluesky_theme} | {eo_theme}"
        sub = df[(df["bluesky_theme"] == bluesky_theme) & (df["eo_theme"] == eo_theme)].sort_values("day").copy()
        status = "tested" if pair_id in tested_pairs else "skipped_manual_review" if pair_id in skipped_pairs else "not_primary_tested"

        sub["pair_id"] = pair_id
        sub["pair_status"] = status
        out_rows.append(sub)

        fig, ax = plt.subplots(figsize=(15, 6.5))
        ax.plot(
            sub["day"],
            sub["eo_count_stationary"],
            color=eo_color,
            linewidth=2.2,
            alpha=0.95,
        )
        ax.axhline(0, color="#888888", linewidth=0.8, alpha=0.6)
        ax.set_ylabel(f"EO stationary ({sub['eo_count_transform_used'].iloc[0]})", color=eo_color)
        ax.tick_params(axis="y", labelcolor=eo_color)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(
            f"{i}. {bluesky_theme}\n{eo_theme}",
            fontsize=16,
            loc="left",
        )

        ax2 = ax.twinx()
        ax2.plot(
            sub["day"],
            sub["bluesky_posts_stationary"],
            color=bs_color,
            linewidth=1.9,
            alpha=0.9,
        )
        ax2.set_ylabel(f"Bluesky stationary ({sub['bluesky_posts_transform_used'].iloc[0]})", color=bs_color)
        ax2.tick_params(axis="y", labelcolor=bs_color)

        subtitle = (
            f"Pair status: {status.replace('_', ' ')} | "
            f"EO transform: {sub['eo_count_transform_used'].iloc[0]} | "
            f"Bluesky transform: {sub['bluesky_posts_transform_used'].iloc[0]}"
        )
        ax.text(
            0.0,
            1.02,
            subtitle,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#555555",
        )

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", rotation=30)
        handles = [
            Line2D([0], [0], color=eo_color, lw=2.2, label=f"EO stationary ({sub['eo_count_transform_used'].iloc[0]})"),
            Line2D([0], [0], color=bs_color, lw=1.9, label=f"Bluesky stationary ({sub['bluesky_posts_transform_used'].iloc[0]})"),
        ]
        ax.legend(handles=handles, loc="upper right", frameon=True)
        fig.tight_layout()

        stem = f"{i:02d}_{slugify(bluesky_theme)}__to__{slugify(eo_theme)}_stationary"
        png_path = OUT_DIR / f"{stem}.png"
        pdf_path = OUT_DIR / f"{stem}.pdf"
        fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
        fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        manifest_rows.append(
            {
                "index": i,
                "pair_id": pair_id,
                "bluesky_theme": bluesky_theme,
                "eo_theme": eo_theme,
                "pair_status": status,
                "eo_count_transform_used": sub["eo_count_transform_used"].iloc[0],
                "bluesky_posts_transform_used": sub["bluesky_posts_transform_used"].iloc[0],
                "png_file": png_path.name,
                "pdf_file": pdf_path.name,
            }
        )

    pd.concat(out_rows, ignore_index=True).to_csv(OUT_DIR / "per_pair_stationary_series_data.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "per_pair_stationary_series_manifest.csv", index=False)

    print(f"folder\t{OUT_DIR}")
    print(f"manifest\t{OUT_DIR / 'per_pair_stationary_series_manifest.csv'}")
    print(f"data\t{OUT_DIR / 'per_pair_stationary_series_data.csv'}")


if __name__ == "__main__":
    main()
