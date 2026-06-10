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


BASE_DIR = Path("local_data/03_Github_data/06_EO_effect_bluesky")
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


def fig_overall_monthly() -> None:
    eo = load("eo_monthly_overall.csv")
    bs = load("bluesky_monthly_overall.csv")
    eo["month"] = pd.to_datetime(eo["month"])
    bs["month"] = pd.to_datetime(bs["month"])
    fig, ax1 = plt.subplots(figsize=(15, 6))
    ax2 = ax1.twinx()
    ax1.plot(bs["month"], bs["bluesky_posts"], color="#1D4ED8", linewidth=2.8, label="Bluesky posts")
    ax2.bar(eo["month"], eo["eo_count"], width=20, color="#EA580C", alpha=0.35, label="EO count")
    ax1.set_title("Overall Monthly Bluesky Volume and Executive Orders")
    ax1.set_ylabel("Bluesky posts")
    ax2.set_ylabel("EO count")
    ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
    save(fig, "overall_monthly_bluesky_vs_eo")


def fig_eo_theme_counts() -> None:
    df = load("eo_theme_counts.csv").sort_values("eo_count", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df["eo_theme"], df["eo_count"], color=sns.color_palette("magma", n_colors=len(df)), edgecolor="#0F172A", linewidth=0.4)
    ax.set_title("Executive Orders by EO Theme")
    ax.set_xlabel("EO count")
    ax.set_ylabel("")
    for p in ax.patches:
        ax.text(p.get_width() + 0.2, p.get_y() + p.get_height() / 2, f"{int(p.get_width())}", va="center", fontsize=9)
    save(fig, "eo_theme_counts_barh")


def fig_crosswalk_theme_comparison() -> None:
    eo = load("eo_theme_counts.csv")
    bs = load("bluesky_theme_counts.csv")
    merged = eo.groupby("bluesky_theme", as_index=False)["eo_count"].sum().merge(bs[["bluesky_theme", "bluesky_posts"]], on="bluesky_theme", how="left").dropna()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(merged["eo_count"], merged["bluesky_posts"], s=180, color="#0F766E", alpha=0.9, edgecolor="white", linewidth=1.2)
    for _, row in merged.iterrows():
        ax.text(row["eo_count"] + 0.3, row["bluesky_posts"], row["bluesky_theme"][:26], fontsize=9)
    ax.set_title("Mapped EO Theme Volume vs Bluesky Theme Volume")
    ax.set_xlabel("EO count")
    ax.set_ylabel("Bluesky posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    save(fig, "mapped_theme_eo_vs_bluesky_scatter")


def fig_theme_monthly_overlay() -> None:
    eo = load("eo_theme_monthly.csv")
    bs = load("bluesky_theme_monthly.csv")
    eo["month"] = pd.to_datetime(eo["month"])
    bs["month"] = pd.to_datetime(bs["month"])
    top = load("eo_theme_counts.csv").head(6)["bluesky_theme"].dropna().unique().tolist()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12), sharex=True)
    axes = axes.reshape(-1)
    for ax, theme in zip(axes, top):
        eo_sub = eo[eo["bluesky_theme"] == theme].groupby("month", as_index=False)["eo_count"].sum()
        bs_sub = bs[bs["bluesky_theme"] == theme].groupby("month", as_index=False)["bluesky_posts"].sum()
        if bs_sub["bluesky_posts"].max() > 0:
            bs_sub["norm_bluesky"] = bs_sub["bluesky_posts"] / bs_sub["bluesky_posts"].max()
        else:
            bs_sub["norm_bluesky"] = 0
        if len(eo_sub) and eo_sub["eo_count"].max() > 0:
            eo_sub["norm_eo"] = eo_sub["eo_count"] / eo_sub["eo_count"].max()
        else:
            eo_sub["norm_eo"] = 0
        ax.plot(bs_sub["month"], bs_sub["norm_bluesky"], color="#2563EB", linewidth=2.4, label="Bluesky (norm)")
        ax.bar(eo_sub["month"], eo_sub["norm_eo"], width=20, color="#EA580C", alpha=0.35, label="EOs (norm)")
        ax.set_title(theme[:55], fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Top Mapped Themes: Normalized Monthly EO vs Bluesky Volume", y=0.995, fontsize=18, fontweight="bold")
    for ax in axes[len(top):]:
        ax.axis("off")
    save(fig, "top_mapped_themes_monthly_overlay")


def fig_event_study_eo_to_bluesky() -> None:
    df = load("event_study_eo_to_bluesky.csv")
    top = df.groupby("theme")["num_events"].max().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), sharex=True)
    axes = axes.reshape(-1)
    for ax, theme in zip(axes, top):
        sub = df[df["theme"] == theme]
        ax.plot(sub["relative_day"], sub["avg_bluesky_posts"], color="#1D4ED8", linewidth=2.3)
        ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.3)
        ax.set_title(theme[:55], fontsize=11)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Average Bluesky Response Around EO Signing Dates", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "event_study_eo_to_bluesky")


def fig_event_study_bluesky_to_eo() -> None:
    df = load("event_study_bluesky_spikes_to_eo.csv")
    top = df.groupby("theme")["num_spikes"].max().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15, 12), sharex=True)
    axes = axes.reshape(-1)
    for ax, theme in zip(axes, top):
        sub = df[df["theme"] == theme]
        ax.plot(sub["relative_day"], sub["avg_eo_count"], color="#EA580C", linewidth=2.3)
        ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.3)
        ax.set_title(theme[:55], fontsize=11)
    for ax in axes[len(top):]:
        ax.axis("off")
    fig.suptitle("Average EO Activity Around Bluesky Theme Spikes", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "event_study_bluesky_spikes_to_eo")


def fig_crosscorr_heatmap() -> None:
    df = load("theme_cross_correlations.csv")
    pivot = df.pivot(index="theme", columns="lag_days", values="correlation")
    order = df.groupby("theme")["correlation"].apply(lambda s: s.abs().max()).sort_values(ascending=False).index
    pivot = pivot.loc[order]
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, cmap="coolwarm", center=0, linewidths=0.15, linecolor="white", ax=ax)
    ax.set_title("Lead-Lag Correlation Between EO Counts and Bluesky Posts")
    ax.set_xlabel("Lag in days (positive = Bluesky follows EO)")
    ax.set_ylabel("")
    save(fig, "eo_bluesky_crosscorr_heatmap")


def fig_crosscorr_peaks() -> None:
    df = load("theme_cross_correlation_peaks.csv").sort_values("correlation", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#2563EB" if x >= 0 else "#DC2626" for x in df["correlation"]]
    ax.barh(df["theme"], df["correlation"], color=colors, edgecolor="#0F172A", linewidth=0.4)
    ax.set_title("Peak Lead-Lag Correlation by Theme")
    ax.set_xlabel("Peak correlation")
    ax.set_ylabel("")
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(row["correlation"] + (0.01 if row["correlation"] >= 0 else -0.01), i, f"lag {int(row['lag_days'])}", va="center", ha="left" if row["correlation"] >= 0 else "right", fontsize=9)
    save(fig, "eo_bluesky_crosscorr_peaks")


def fig_granger_heatmap() -> None:
    df = load("theme_granger_results.csv")
    melted = df.melt(id_vars="theme", value_vars=["eo_to_bluesky_min_p", "bluesky_to_eo_min_p"], var_name="direction", value_name="p_value")
    pivot = melted.pivot(index="theme", columns="direction", values="p_value")
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(pivot, cmap="viridis_r", linewidths=0.2, linecolor="white", annot=True, fmt=".3f", ax=ax)
    ax.set_title("Minimum Granger p-values by Theme")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "granger_pvalue_heatmap")


def fig_top5_eo_theme_stacked() -> None:
    eo = load("eo_theme_monthly.csv")
    top = load("eo_theme_counts.csv").head(5)["eo_theme"].tolist()
    eo["month"] = pd.to_datetime(eo["month"])
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 18), sharex=True)
    for ax, theme in zip(axes, top):
        sub = eo[eo["eo_theme"] == theme].copy()
        ax.bar(sub["month"], sub["eo_count"], width=20, color="#EA580C", alpha=0.75)
        ax.set_title(theme, loc="left", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    fig.suptitle("Top 5 EO Themes by Monthly Signing Volume", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "top5_eo_themes_monthly")


def main() -> None:
    ensure_dirs()
    set_theme()
    fig_overall_monthly()
    fig_eo_theme_counts()
    fig_crosswalk_theme_comparison()
    fig_theme_monthly_overlay()
    fig_event_study_eo_to_bluesky()
    fig_event_study_bluesky_to_eo()
    fig_crosscorr_heatmap()
    fig_crosscorr_peaks()
    fig_granger_heatmap()
    fig_top5_eo_theme_stacked()
    manifest = sorted(p.name for p in FIG_DIR.iterdir() if p.is_file())
    (FIG_DIR / "figure_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"figure_dir\t{FIG_DIR}")
    print(f"saved_files\t{len(manifest)}")


if __name__ == "__main__":
    main()
