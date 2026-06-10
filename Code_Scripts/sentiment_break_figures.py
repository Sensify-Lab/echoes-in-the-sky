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
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path("local_data/03_Github_data/05_bluesky_data_merged_sentiment/theme_sentiment_break_analysis")
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


def fmt_int(value: float, _pos: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.0f}K"
    return f"{value:.0f}"


COLORS = {"negative": "#B91C1C", "neutral": "#64748B", "positive": "#15803D"}


def fig_monthly_stacked(monthly_share: pd.DataFrame) -> None:
    df = monthly_share.copy()
    df["month"] = pd.to_datetime(df["month"], utc=True).dt.tz_convert(None)
    pivot_counts = df.pivot(index="month", columns="sentiment_label", values="posts").fillna(0)
    pivot_share = df.pivot(index="month", columns="sentiment_label", values="pct_posts").fillna(0)
    order = [c for c in ["negative", "neutral", "positive"] if c in pivot_counts.columns]
    pivot_counts = pivot_counts[order]
    pivot_share = pivot_share[order]
    fig, ax = plt.subplots(figsize=(16, 7))
    bottom = np.zeros(len(pivot_counts))
    for col in order:
        vals = pivot_counts[col].to_numpy()
        ax.bar(pivot_counts.index, vals, bottom=bottom, width=25, color=COLORS[col], label=col)
        centers = bottom + vals / 2
        shares = pivot_share[col].to_numpy()
        for x, y, pct, v in zip(pivot_counts.index, centers, shares, vals):
            if pct >= 8:
                ax.text(x, y, f"{pct:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        bottom += vals
    ax.set_title("Monthly Bluesky Posts by Text Sentiment Label")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(title="")
    save(fig, "monthly_posts_stacked_sentiment")


def fig_theme_distribution(theme_share: pd.DataFrame) -> None:
    df = theme_share.copy()
    order = df.groupby("theme_name")["posts"].sum().sort_values(ascending=True).index
    pivot_posts = df.pivot(index="theme_name", columns="sentiment_label", values="posts").fillna(0).loc[order]
    pivot_pct = df.pivot(index="theme_name", columns="sentiment_label", values="pct_posts").fillna(0).loc[order]
    cols = [c for c in ["negative", "neutral", "positive"] if c in pivot_posts.columns]
    fig, ax = plt.subplots(figsize=(14, 9))
    left = np.zeros(len(pivot_posts))
    for col in cols:
        vals = pivot_posts[col].to_numpy()
        ax.barh(pivot_posts.index, vals, left=left, color=COLORS[col], label=col)
        for i, (lft, val, pct) in enumerate(zip(left, vals, pivot_pct[col].to_numpy())):
            if pct >= 10:
                ax.text(lft + val / 2, i, f"{pct:.0f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
        left += vals
    ax.set_title("Overarching Themes by Text Sentiment Label")
    ax.set_xlabel("Posts")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.legend(title="")
    save(fig, "theme_distribution_stacked_sentiment")


def fig_theme_small_multiples(theme_monthly: pd.DataFrame) -> None:
    df = theme_monthly.copy()
    df["month"] = pd.to_datetime(df["month"], utc=True).dt.tz_convert(None)
    themes = df.groupby("theme_name")["posts"].sum().sort_values(ascending=False).index.tolist()
    n = len(themes)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 3.7 * nrows), sharex=True, sharey=False)
    axes = np.array(axes).reshape(-1)
    for ax, theme in zip(axes, themes):
        sub = df[df["theme_name"] == theme].copy()
        pivot = sub.pivot(index="month", columns="sentiment_label", values="posts").fillna(0)
        order = [c for c in ["negative", "neutral", "positive"] if c in pivot.columns]
        for col in order:
            ax.plot(pivot.index, pivot[col], label=col, color=COLORS[col], linewidth=1.8)
        ax.set_title(theme[:58], fontsize=10)
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("Temporal Sentiment Trends for Each Theme", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "all_theme_temporal_sentiment_small_multiples")


def fig_top5_panel(theme_monthly: pd.DataFrame, top5: pd.DataFrame) -> None:
    df = theme_monthly.copy()
    df["month"] = pd.to_datetime(df["month"], utc=True).dt.tz_convert(None)
    keep = top5["theme_name"].tolist()
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 18), sharex=True)
    for ax, theme in zip(axes, keep):
        sub = df[df["theme_name"] == theme]
        pivot = sub.pivot(index="month", columns="sentiment_label", values="posts").fillna(0)
        order = [c for c in ["negative", "neutral", "positive"] if c in pivot.columns]
        bottom = np.zeros(len(pivot))
        for col in order:
            vals = pivot[col].to_numpy()
            ax.bar(pivot.index, vals, bottom=bottom, width=25, color=COLORS[col], label=col)
            bottom += vals
        ax.set_title(theme, loc="left", fontsize=11, fontweight="bold")
        ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[-1].get_xticklabels(), rotation=45, ha="right")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Top 5 Themes: Monthly Post Volume by Sentiment", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "top5_themes_monthly_stacked_sentiment")


def fig_break_scores(breaks: pd.DataFrame) -> None:
    df = breaks.head(15).sort_values("break_score", ascending=True)
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(df["theme_name"], df["break_score"], color=sns.color_palette("magma", n_colors=len(df)), edgecolor="#0F172A", linewidth=0.4)
    ax.set_title("Estimated Structural Break Strength by Theme")
    ax.set_xlabel("Break score")
    ax.set_ylabel("")
    for p in ax.patches:
        ax.text(p.get_width() + 0.03, p.get_y() + p.get_height() / 2, f"{p.get_width():.2f}", va="center", fontsize=9)
    save(fig, "theme_structural_break_scores")


def fig_break_scatter(breaks: pd.DataFrame) -> None:
    df = breaks.copy()
    df["volume_shift"] = df["post_total_posts_avg"] - df["pre_total_posts_avg"]
    df["negative_share_shift"] = df["post_negative_share_avg"] - df["pre_negative_share_avg"]
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(df["volume_shift"], df["negative_share_shift"], s=np.clip(df["break_score"] * 70, 70, 900), c=df["break_score"], cmap="viridis", alpha=0.9, edgecolor="white", linewidth=1.0)
    for _, row in df.iterrows():
        ax.text(row["volume_shift"], row["negative_share_shift"], row["theme_name"][:24], fontsize=8, ha="left", va="bottom")
    ax.axhline(0, color="#94A3B8", linewidth=1)
    ax.axvline(0, color="#94A3B8", linewidth=1)
    ax.set_title("Theme Structural Break Map")
    ax.set_xlabel("Post-break minus pre-break volume average")
    ax.set_ylabel("Post-break minus pre-break negative-share average")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Break score")
    save(fig, "theme_structural_break_scatter")


def fig_break_small_multiples(theme_totals: pd.DataFrame, breaks: pd.DataFrame) -> None:
    df = theme_totals.copy()
    df["month"] = pd.to_datetime(df["month"], utc=True).dt.tz_convert(None)
    top_breaks = breaks.head(6)["theme_name"].tolist()
    fig = plt.figure(figsize=(16, 16))
    gs = GridSpec(6, 2, figure=fig, width_ratios=[2.2, 1.2], hspace=0.45, wspace=0.18)
    for i, theme in enumerate(top_breaks):
        sub = df[df["theme_name"] == theme].sort_values("month")
        br = breaks[breaks["theme_name"] == theme].iloc[0]
        break_month = pd.to_datetime(br["break_month"])
        ax1 = fig.add_subplot(gs[i, 0])
        ax2 = fig.add_subplot(gs[i, 1], sharex=ax1)
        ax1.plot(sub["month"], sub["total_posts"], color="#1D4ED8", linewidth=2.2)
        ax1.axvline(break_month, color="#DC2626", linestyle="--", linewidth=1.5)
        ax1.set_title(theme[:70], fontsize=11, loc="left", fontweight="bold")
        ax1.yaxis.set_major_formatter(FuncFormatter(fmt_int))
        ax1.set_ylabel("Posts")
        ax2.plot(sub["month"], sub["negative_share_pct"], color="#B91C1C", linewidth=2.0)
        ax2.axvline(break_month, color="#DC2626", linestyle="--", linewidth=1.5)
        ax2.set_ylabel("Neg %")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
        for ax in (ax1, ax2):
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.suptitle("Structural Break Diagnostics for Highest-Break Themes", y=0.995, fontsize=18, fontweight="bold")
    save(fig, "theme_structural_break_diagnostics")


def main() -> None:
    ensure_dirs()
    set_theme()
    fig_monthly_stacked(load("sentiment_by_month_share.csv"))
    fig_theme_distribution(load("theme_sentiment_share.csv"))
    theme_monthly = load("theme_monthly_sentiment.csv")
    fig_theme_small_multiples(theme_monthly)
    fig_top5_panel(theme_monthly, load("top5_themes.csv"))
    breaks = load("theme_structural_breaks.csv")
    fig_break_scores(breaks)
    fig_break_scatter(breaks)
    fig_break_small_multiples(load("theme_monthly_totals.csv"), breaks)
    manifest = sorted(p.name for p in FIG_DIR.iterdir() if p.is_file())
    (FIG_DIR / "figure_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"figure_dir\t{FIG_DIR}")
    print(f"saved_files\t{len(manifest)}")


if __name__ == "__main__":
    main()
