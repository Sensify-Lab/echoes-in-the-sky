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


BASE_DIR = Path("local_data/03_Github_data/05_bluesky_data_merged_sentiment/text_sentiment_analysis_outputs")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"


def ensure_dirs() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str) -> pd.DataFrame:
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


def fig_sentiment_distribution(df: pd.DataFrame) -> None:
    order = ["negative", "neutral", "positive"]
    df["text_clean_sentimentlabel"] = pd.Categorical(df["text_clean_sentimentlabel"], categories=order, ordered=True)
    df = df.sort_values("text_clean_sentimentlabel")
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#B91C1C", "#64748B", "#15803D"]
    ax.bar(df["text_clean_sentimentlabel"], df["rows"], color=colors, edgecolor="#0F172A", linewidth=0.6)
    ax.set_title("Distribution of Text Sentiment Labels")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    for p, pct in zip(ax.patches, df["pct_rows"]):
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + df["rows"].max() * 0.015, f"{pct:.1f}%", ha="center", fontsize=11)
    save(fig, "text_sentiment_distribution")


def fig_sentiment_scores(df: pd.DataFrame) -> None:
    order = ["negative", "neutral", "positive"]
    df["text_clean_sentimentlabel"] = pd.Categorical(df["text_clean_sentimentlabel"], categories=order, ordered=True)
    df = df.sort_values("text_clean_sentimentlabel")
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["text_clean_sentimentlabel"], df["avg_score"], color=["#DC2626", "#94A3B8", "#16A34A"], edgecolor="#0F172A", linewidth=0.6)
    ax.set_title("Average Sentiment Score by Label")
    ax.set_xlabel("")
    ax.set_ylabel("Average Score")
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + (0.02 if p.get_height() >= 0 else -0.05), f"{p.get_height():.3f}", ha="center", fontsize=11)
    save(fig, "text_sentiment_avg_scores")


def fig_theme_heatmap(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="theme_name", columns="text_clean_sentimentlabel", values="rows").fillna(0)
    order = df.groupby("theme_name")["rows"].sum().sort_values(ascending=False).index
    pivot = pivot.loc[order]
    pivot = pivot[[c for c in ["negative", "neutral", "positive"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.3, linecolor="white", cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    ax.set_title("Theme by Text Sentiment Label")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "theme_by_text_sentiment_heatmap")


def fig_theme_share(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="theme_name", columns="text_clean_sentimentlabel", values="pct_within_theme").fillna(0)
    order = df.groupby("theme_name")["rows"].sum().sort_values(ascending=False).index
    pivot = pivot.loc[order]
    pivot = pivot[[c for c in ["negative", "neutral", "positive"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(pivot, cmap="crest", linewidths=0.3, linecolor="white", annot=True, fmt=".1f", ax=ax)
    ax.set_title("Within-Theme Sentiment Share (%)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, "theme_sentiment_share_heatmap")


def fig_monthly_counts(df: pd.DataFrame) -> None:
    df["month"] = pd.to_datetime(df["month"], utc=True).dt.tz_convert(None)
    fig, ax = plt.subplots(figsize=(15, 6))
    palette = {"negative": "#B91C1C", "neutral": "#64748B", "positive": "#15803D"}
    for label, sub in df.groupby("text_clean_sentimentlabel"):
        sub = sub.sort_values("month")
        ax.plot(sub["month"], sub["rows"], label=label, linewidth=2.5, color=palette.get(label))
    ax.set_title("Monthly Text Sentiment Volume")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    save(fig, "monthly_text_sentiment_lines")


def fig_monthly_share(df: pd.DataFrame) -> None:
    df["month"] = pd.to_datetime(df["month"], utc=True).dt.tz_convert(None)
    pivot = df.pivot(index="month", columns="text_clean_sentimentlabel", values="pct_within_month").fillna(0)
    pivot = pivot[[c for c in ["negative", "neutral", "positive"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(15, 6))
    colors = ["#B91C1C", "#64748B", "#15803D"]
    ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, colors=colors, alpha=0.95)
    ax.set_title("Monthly Text Sentiment Share")
    ax.set_xlabel("")
    ax.set_ylabel("Percent of Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend()
    save(fig, "monthly_text_sentiment_share_area")


def fig_quarterly(df: pd.DataFrame) -> None:
    df["quarter"] = pd.to_datetime(df["quarter"], utc=True).dt.tz_convert(None)
    pivot = df.pivot(index="quarter", columns="text_clean_sentimentlabel", values="rows").fillna(0)
    pivot = pivot[[c for c in ["negative", "neutral", "positive"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(13, 6))
    bottom = np.zeros(len(pivot))
    colors = {"negative": "#B91C1C", "neutral": "#64748B", "positive": "#15803D"}
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col], bottom=bottom, label=col, color=colors[col], width=55)
        bottom += pivot[col].to_numpy()
    ax.set_title("Quarterly Text Sentiment Volume")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-Q%q"))
    ax.legend()
    save(fig, "quarterly_text_sentiment_stacked")


def fig_engagement(df: pd.DataFrame) -> None:
    order = ["negative", "neutral", "positive"]
    metrics = ["avg_likes", "avg_replies", "avg_reposts", "avg_quotes"]
    melted = df.melt(id_vars="text_clean_sentimentlabel", value_vars=metrics, var_name="metric", value_name="value")
    melted["text_clean_sentimentlabel"] = pd.Categorical(melted["text_clean_sentimentlabel"], categories=order, ordered=True)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=melted, x="metric", y="value", hue="text_clean_sentimentlabel", palette=["#B91C1C", "#64748B", "#15803D"], ax=ax)
    ax.set_title("Average Engagement by Text Sentiment Label")
    ax.set_xlabel("")
    ax.set_ylabel("Average count")
    ax.legend(title="")
    save(fig, "engagement_by_text_sentiment")


def fig_content_features(df: pd.DataFrame) -> None:
    rows_map = dict(zip(df["text_clean_sentimentlabel"], df["rows"]))
    feat = df.copy()
    for col in ["external_links", "image_embeds", "media_image_embeds", "video_embeds", "hashtagged_posts"]:
        feat[col] = 100.0 * feat[col] / feat["rows"]
    melted = feat.melt(
        id_vars="text_clean_sentimentlabel",
        value_vars=["external_links", "image_embeds", "media_image_embeds", "video_embeds", "hashtagged_posts"],
        var_name="feature",
        value_name="pct",
    )
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=melted, x="feature", y="pct", hue="text_clean_sentimentlabel", palette=["#B91C1C", "#64748B", "#15803D"], ax=ax)
    ax.set_title("Content Features by Text Sentiment Label")
    ax.set_xlabel("")
    ax.set_ylabel("Share of posts (%)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="")
    save(fig, "content_features_by_text_sentiment")


def fig_media_match(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="text_clean_sentimentlabel", columns="media_merge_clean_sentimentlabel", values="rows").fillna(0)
    pivot = pivot.reindex(index=["negative", "neutral", "positive"], columns=["negative", "neutral", "positive"])
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot, cmap="mako", linewidths=0.4, linecolor="white", annot=True, fmt=".0f", cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    ax.set_title("Text vs Media Sentiment Label Agreement")
    ax.set_xlabel("Media-clean sentiment label")
    ax.set_ylabel("Text-clean sentiment label")
    save(fig, "text_vs_media_sentiment_heatmap")


def fig_theme_mismatch(df: pd.DataFrame) -> None:
    df = df.sort_values("mismatch_pct", ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(df["theme_name"], df["mismatch_pct"], color=sns.color_palette("rocket", n_colors=len(df)), edgecolor="#0F172A", linewidth=0.4)
    ax.set_title("Themes with Highest Text-vs-Media Sentiment Mismatch")
    ax.set_xlabel("Mismatch rate (%)")
    ax.set_ylabel("")
    for p in ax.patches:
        ax.text(p.get_width() + 0.05, p.get_y() + p.get_height() / 2, f"{p.get_width():.2f}%", va="center", fontsize=10)
    save(fig, "theme_mismatch_rate_barh")


def fig_partition(df: pd.DataFrame) -> None:
    pivot = df.pivot(index="data_partition", columns="text_clean_sentimentlabel", values="rows").fillna(0)
    pivot = pivot[[c for c in ["negative", "neutral", "positive"] if c in pivot.columns]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(pivot))
    colors = {"negative": "#B91C1C", "neutral": "#64748B", "positive": "#15803D"}
    for col in pivot.columns:
        ax.bar(pivot.index, pivot[col], bottom=bottom, label=col, color=colors[col])
        bottom += pivot[col].to_numpy()
    ax.set_title("Text Sentiment by Data Partition")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.legend()
    save(fig, "partition_text_sentiment_stacked")


def main() -> None:
    ensure_dirs()
    set_theme()
    fig_sentiment_distribution(load_csv("sentiment_label_distribution.csv"))
    fig_sentiment_scores(load_csv("sentiment_score_summary.csv"))
    fig_theme_heatmap(load_csv("theme_sentiment_distribution.csv"))
    fig_theme_share(load_csv("theme_sentiment_share.csv"))
    fig_monthly_counts(load_csv("monthly_sentiment_counts.csv"))
    fig_monthly_share(load_csv("monthly_sentiment_share.csv"))
    fig_quarterly(load_csv("quarterly_sentiment_counts.csv"))
    fig_engagement(load_csv("engagement_by_sentiment.csv"))
    fig_content_features(load_csv("content_features_by_sentiment.csv"))
    fig_media_match(load_csv("media_label_match_by_text_label.csv"))
    fig_theme_mismatch(load_csv("theme_mismatch_counts.csv"))
    fig_partition(load_csv("partition_sentiment_distribution.csv"))
    manifest = sorted(p.name for p in FIG_DIR.iterdir() if p.is_file())
    (FIG_DIR / "figure_manifest.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")
    print(f"saved_figures\t{len(manifest)}")
    print(f"figure_dir\t{FIG_DIR}")


if __name__ == "__main__":
    main()
