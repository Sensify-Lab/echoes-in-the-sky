from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SOURCE_SCRIPT = Path(__file__).with_name("structure_break_fixed_ci.py")
OUTPUT_BASE = Path(
    "local_data/03_Github_data/08_EO_effect_structure_changes/"
    "post_2025_01_20_max20_exact_dates_warm_segments"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location("structure_break_base", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base_module()
base.BASE_DIR = OUTPUT_BASE
base.TABLE_DIR = OUTPUT_BASE / "tables"
base.FIG_DIR = OUTPUT_BASE / "publication_figures"
base.PER_THEME_DIR = base.FIG_DIR / "per_theme"


def plot_theme(theme: str, sub: pd.DataFrame, breaks_df: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="talk")
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

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(sub["day"], sub["posts"], color="#1f77b4", linewidth=1.8, label="Daily posts", zorder=3)
    ax.set_title(f"{theme}: Daily Posts After January 20, 2025", loc="left", fontsize=18)
    ax.set_ylabel("Posts per day")
    ax.set_xlabel("Month-Year")
    ax.grid(False)

    segment_bounds = [sub["day"].iloc[0]]
    segment_bounds.extend(pd.to_datetime(breaks_df["Break_date"]).tolist())
    segment_bounds.append(sub["day"].iloc[-1])
    warm_colors = ["#fff1cc", "#ffd6a5", "#ffb4a2", "#f4a261"]
    for i in range(len(segment_bounds) - 1):
        left = segment_bounds[i]
        right = segment_bounds[i + 1]
        ax.axvspan(left, right, color=warm_colors[i % len(warm_colors)], alpha=0.42, zorder=0)

    ymax = max(float(sub["posts"].max()), 1.0)
    label_levels = [0.97, 0.91, 0.85]
    for idx, (_, row) in enumerate(breaks_df.iterrows()):
        ci_low = pd.to_datetime(row["CI_2.5_date"])
        brk = pd.to_datetime(row["Break_date"])
        ci_high = pd.to_datetime(row["CI_97.5_date"])

        ci_left = ci_low
        ci_right = ci_high
        if ci_left == ci_right:
            ci_left = ci_left - pd.Timedelta(days=0.75)
            ci_right = ci_right + pd.Timedelta(days=0.75)

        label_y = ymax * label_levels[idx % len(label_levels)]
        label_y_mid = max(label_y - ymax * 0.06, ymax * 0.68)

        ax.axvspan(ci_left, ci_right, color="#c0392b", alpha=0.12, zorder=1)
        ax.axvline(ci_low, color="#8e2a1e", linewidth=1.8, linestyle="-", zorder=4)
        ax.axvline(brk, color="#6e0000", linewidth=2.5, linestyle="--", zorder=5)
        ax.axvline(ci_high, color="#8e2a1e", linewidth=1.8, linestyle="-", zorder=4)

        ax.text(
            ci_low,
            label_y,
            f"CI low\n{ci_low.strftime('%Y-%m-%d')}",
            color="#6e0000",
            fontsize=8,
            ha="right",
            va="top",
        )
        ax.text(
            brk,
            label_y_mid,
            f"Break\n{brk.strftime('%Y-%m-%d')}",
            color="#6e0000",
            fontsize=8,
            ha="center",
            va="top",
        )
        ax.text(
            ci_high,
            label_y,
            f"CI high\n{ci_high.strftime('%Y-%m-%d')}",
            color="#6e0000",
            fontsize=8,
            ha="left",
            va="top",
        )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    stem = base.slugify(theme)
    fig.savefig(
        base.PER_THEME_DIR / f"{stem}_breaks_post_2025_01_20_max20_exact_dates_warm_segments.png",
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        base.PER_THEME_DIR / f"{stem}_breaks_post_2025_01_20_max20_exact_dates_warm_segments.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


base.plot_theme = plot_theme


if __name__ == "__main__":
    base.main()
