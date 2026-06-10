from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path("local_data/03_Github_data/07_EO_effect_bluesky_2025_01_20_to_2026_02_01")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"


def ensure_dirs() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fmt_int(value: float, _pos: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.0f}K"
    return f"{value:.0f}"


def set_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#334155",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0F172A",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def fig_overall_window() -> None:
    eo = load("eo_monthly_overall.csv")
    bs = load("bluesky_monthly_overall.csv")
    eo["month"] = pd.to_datetime(eo["month"])
    bs["month"] = pd.to_datetime(bs["month"])
    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()
    ax1.plot(bs["month"], bs["bluesky_posts"], color="#2563EB", linewidth=2.8)
    ax2.bar(eo["month"], eo["eo_count"], width=18, color="#EA580C", alpha=0.35)
    ax1.set_title("Windowed Monthly Bluesky Volume and EO Count")
    ax1.set_ylabel("Bluesky posts")
    ax2.set_ylabel("EO count")
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    save(fig, "windowed_monthly_bluesky_vs_eo")


def fig_theme_overlay() -> None:
    eo = load("eo_theme_monthly.csv")
    bs = load("bluesky_theme_monthly.csv")
    top = load("eo_theme_counts.csv").head(6)["bluesky_theme"].dropna().unique().tolist()
    eo["month"] = pd.to_datetime(eo["month"])
    bs["month"] = pd.to_datetime(bs["month"])
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    axes = axes.reshape(-1)
    for ax, theme in zip(axes, top):
        eo_sub = eo[eo["bluesky_theme"] == theme].groupby("month", as_index=False)["eo_count"].sum()
        bs_sub = bs[bs["bluesky_theme"] == theme].groupby("month", as_index=False)["bluesky_posts"].sum()
        bs_sub["norm_bluesky"] = bs_sub["bluesky_posts"] / (bs_sub["bluesky_posts"].max() or 1)
        eo_sub["norm_eo"] = eo_sub["eo_count"] / (eo_sub["eo_count"].max() or 1)
        ax.plot(bs_sub["month"], bs_sub["norm_bluesky"], color="#2563EB", linewidth=2.2)
        ax.bar(eo_sub["month"], eo_sub["norm_eo"], width=18, color="#EA580C", alpha=0.35)
        ax.set_title(theme[:55], fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Windowed Theme Overlays: EO vs Bluesky", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "windowed_theme_monthly_overlay")


def fig_event_eo_to_bluesky() -> None:
    df = load("event_study_eo_to_bluesky.csv")
    top = df.groupby("theme")["num_events"].max().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    axes = axes.reshape(-1)
    for ax, theme in zip(axes, top):
        sub = df[df["theme"] == theme]
        ax.plot(sub["relative_day"], sub["avg_bluesky_posts"], color="#2563EB", linewidth=2.2)
        ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.2)
        ax.set_title(theme[:55], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Windowed EO-Centered Event Study", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "windowed_event_study_eo_to_bluesky")


def fig_event_bluesky_to_eo() -> None:
    df = load("event_study_bluesky_spikes_to_eo.csv")
    top = df.groupby("theme")["num_spikes"].max().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(3, 2, figsize=(15, 11), sharex=True)
    axes = axes.reshape(-1)
    for ax, theme in zip(axes, top):
        sub = df[df["theme"] == theme]
        ax.plot(sub["relative_day"], sub["avg_eo_count"], color="#EA580C", linewidth=2.2)
        ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.2)
        ax.set_title(theme[:55], fontsize=10)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Windowed Discourse-Spike-Centered EO Activity", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "windowed_event_study_bluesky_to_eo")


def fig_crosscorr_heatmap() -> None:
    df = load("theme_cross_correlations.csv")
    pivot = df.pivot(index="theme", columns="lag_days", values="correlation")
    order = df.groupby("theme")["correlation"].apply(lambda s: s.abs().max()).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(pivot, cmap="coolwarm", center=0, linewidths=0.12, linecolor="white", ax=ax)
    ax.set_title("Windowed EO vs Bluesky Cross-Correlation")
    ax.set_xlabel("Lag in days (positive = Bluesky follows EO)")
    ax.set_ylabel("")
    save(fig, "windowed_crosscorr_heatmap")


def fig_granger() -> None:
    df = load("theme_granger_results.csv")
    if df.empty:
        return
    melted = df.melt(id_vars="theme", value_vars=["eo_to_bluesky_min_p", "bluesky_to_eo_min_p"], var_name="direction", value_name="p_value")
    pivot = melted.pivot(index="theme", columns="direction", values="p_value")
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, cmap="viridis_r", linewidths=0.2, linecolor="white", annot=True, fmt=".3f", ax=ax)
    ax.set_title("Windowed Minimum Granger p-values")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "windowed_granger_heatmap")


def main() -> None:
    ensure_dirs()
    set_theme()
    fig_overall_window()
    fig_theme_overlay()
    fig_event_eo_to_bluesky()
    fig_event_bluesky_to_eo()
    fig_crosscorr_heatmap()
    fig_granger()
    manifest = sorted(p.name for p in FIG_DIR.iterdir() if p.is_file())
    (FIG_DIR / "figure_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"figure_dir\t{FIG_DIR}")
    print(f"saved_files\t{len(manifest)}")


if __name__ == "__main__":
    main()
