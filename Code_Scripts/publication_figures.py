from __future__ import annotations

import os
from ast import literal_eval
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import duckdb
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter


BASE_DIR = Path("local_data/03_Github_data/04_final_merged_10M")
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
FIG_DIR = OUTPUT_DIR / "publication_figures"
PARQUET_GLOB = str(BASE_DIR / "*.parquet")


def ensure_dirs() -> None:
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(OUTPUT_DIR / name, **kwargs)


def parse_mixed_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").dt.tz_convert(None)


def save_figure(fig: plt.Figure, name: str) -> None:
    png_path = FIG_DIR / f"{name}.png"
    pdf_path = FIG_DIR / f"{name}.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def set_theme() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "axes.titlesize": 18,
            "axes.titleweight": "bold",
            "axes.labelsize": 13,
            "legend.frameon": False,
            "grid.color": "#D1D5DB",
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fmt_int(value: float, _pos: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.0f}K"
    return f"{value:.0f}"


def annotate_bars(ax: plt.Axes, fmt: str = "{:,.0f}", pct: bool = False) -> None:
    xmax = ax.get_xlim()[1]
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        label = f"{width:.1f}%" if pct else fmt.format(width)
        ax.text(width + xmax * 0.01, y, label, va="center", ha="left", fontsize=10, color="#111827")


def parse_quantiles(value: object) -> list[float]:
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(v) for v in value]
    if isinstance(value, str):
        parsed = literal_eval(value)
        return [float(v) for v in parsed]
    raise ValueError(f"Unsupported quantile value: {value!r}")


def chart_theme_distribution(theme_df: pd.DataFrame) -> None:
    df = theme_df.sort_values("rows", ascending=True)
    palette = sns.color_palette("YlOrBr", n_colors=len(df))
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.barh(df["theme_name"], df["rows"], color=palette, edgecolor="#7C2D12", linewidth=0.6)
    ax.set_title("Theme Distribution Across 10.46M Posts")
    ax.set_xlabel("Posts")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    annotate_bars(ax)
    save_figure(fig, "theme_distribution_barh")


def chart_theme_share(theme_df: pd.DataFrame) -> None:
    df = theme_df.sort_values("pct_rows", ascending=False)
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = sns.color_palette("crest", n_colors=len(df))
    ax.barh(df["theme_name"][::-1], df["pct_rows"][::-1], color=colors[::-1], edgecolor="#0F172A", linewidth=0.5)
    ax.set_title("Theme Share of Total Dataset")
    ax.set_xlabel("Share of Rows (%)")
    ax.set_ylabel("")
    annotate_bars(ax, pct=True)
    save_figure(fig, "theme_share_barh")


def chart_theme_confidence(theme_df: pd.DataFrame) -> None:
    df = theme_df.sort_values("avg_confidence", ascending=True)
    fig, ax = plt.subplots(figsize=(13, 8))
    colors = sns.color_palette("mako", n_colors=len(df))
    ax.barh(df["theme_name"], df["avg_confidence"], color=colors, edgecolor="#1F2937", linewidth=0.5)
    ax.set_title("Average Theme Confidence by Theme")
    ax.set_xlabel("Average Confidence Score")
    ax.set_ylabel("")
    for patch in ax.patches:
        width = patch.get_width()
        y = patch.get_y() + patch.get_height() / 2
        ax.text(width + 0.002, y, f"{width:.3f}", va="center", ha="left", fontsize=10)
    save_figure(fig, "theme_confidence_barh")


def chart_monthly_volume(monthly_df: pd.DataFrame) -> None:
    df = monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["month"], df["rows"], color="#0F766E", linewidth=3)
    ax.fill_between(df["month"], df["rows"], color="#99F6E4", alpha=0.6)
    ax.set_title("Monthly Posting Volume")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save_figure(fig, "monthly_volume_area")


def chart_monthly_volume_log(monthly_df: pd.DataFrame) -> None:
    df = monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df["month"], df["rows"], color="#1D4ED8", linewidth=2.8, marker="o", markersize=4)
    ax.set_yscale("log")
    ax.set_title("Monthly Posting Volume on Log Scale")
    ax.set_xlabel("")
    ax.set_ylabel("Posts (log scale)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save_figure(fig, "monthly_volume_log_line")


def chart_monthly_heatmap(monthly_df: pd.DataFrame) -> None:
    df = monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    df["year"] = df["month"].dt.year
    df["month_name"] = df["month"].dt.strftime("%b")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = df.pivot(index="year", columns="month_name", values="rows").reindex(columns=order)
    fig, ax = plt.subplots(figsize=(14, 4.8))
    sns.heatmap(
        pivot,
        cmap="YlGnBu",
        linewidths=0.5,
        linecolor="white",
        annot=True,
        fmt=".0f",
        cbar_kws={"format": FuncFormatter(fmt_int)},
        ax=ax,
    )
    ax.set_title("Monthly Posting Volume Heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "monthly_volume_heatmap")


def chart_top8_theme_lines(theme_monthly_df: pd.DataFrame) -> None:
    df = theme_monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    fig, ax = plt.subplots(figsize=(15, 7))
    palette = sns.color_palette("tab10", n_colors=df["theme_name"].nunique())
    for color, (theme, sub) in zip(palette, df.groupby("theme_name")):
        sub = sub.sort_values("month")
        ax.plot(sub["month"], sub["rows"], label=theme, linewidth=2.3, color=color)
    ax.set_title("Top 8 Themes by Month")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    save_figure(fig, "top8_theme_monthly_lines")


def chart_top8_theme_stacked_area(theme_monthly_df: pd.DataFrame) -> None:
    df = theme_monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    pivot = df.pivot(index="month", columns="theme_name", values="rows").fillna(0)
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = sns.color_palette("Spectral", n_colors=pivot.shape[1])
    ax.stackplot(pivot.index, pivot.T.values, labels=pivot.columns, colors=colors, alpha=0.95)
    ax.set_title("Composition of Monthly Volume Across Top 8 Themes")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    save_figure(fig, "top8_theme_stacked_area")


def chart_top8_theme_share_area(theme_monthly_df: pd.DataFrame) -> None:
    df = theme_monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    pivot = df.pivot(index="month", columns="theme_name", values="rows").fillna(0)
    share = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = sns.color_palette("Set2", n_colors=share.shape[1])
    ax.stackplot(share.index, share.T.values, labels=share.columns, colors=colors, alpha=0.96)
    ax.set_title("Monthly Theme Share Across Top 8 Themes")
    ax.set_xlabel("")
    ax.set_ylabel("Share")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    save_figure(fig, "top8_theme_share_area")


def chart_top8_theme_heatmap(theme_monthly_df: pd.DataFrame) -> None:
    df = theme_monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"]).dt.strftime("%Y-%m")
    pivot = df.pivot(index="theme_name", columns="month", values="rows").fillna(0)
    ordered_themes = df.groupby("theme_name")["rows"].sum().sort_values(ascending=False).index
    pivot = pivot.loc[ordered_themes]
    fig, ax = plt.subplots(figsize=(16, 7))
    sns.heatmap(pivot, cmap="rocket_r", linewidths=0.2, linecolor="white", cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    ax.set_title("Top 8 Themes by Month Heatmap")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "top8_theme_monthly_heatmap")


def chart_top_languages(top_lang_df: pd.DataFrame) -> None:
    df = top_lang_df.head(12).sort_values("uses", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df["lang"], df["uses"], color=sns.color_palette("viridis", n_colors=len(df)), edgecolor="#111827", linewidth=0.5)
    ax.set_title("Top Language Tags")
    ax.set_xlabel("Tag Uses")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    annotate_bars(ax)
    save_figure(fig, "top_languages_barh")


def chart_top_authors_posts(authors_posts_df: pd.DataFrame) -> None:
    df = authors_posts_df.head(15).copy().sort_values("posts", ascending=True)
    labels = df["author_handle"].fillna(df["author_did"])
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(labels, df["posts"], color=sns.color_palette("flare", n_colors=len(df)), edgecolor="#4C0519", linewidth=0.5)
    ax.set_title("Top Authors by Post Volume")
    ax.set_xlabel("Posts")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    annotate_bars(ax)
    save_figure(fig, "top_authors_by_posts_barh")


def chart_top_authors_engagement(authors_eng_df: pd.DataFrame) -> None:
    df = authors_eng_df.head(15).copy().sort_values("total_engagement", ascending=True)
    labels = df["author_handle"].fillna(df["author_did"])
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.barh(labels, df["total_engagement"], color=sns.color_palette("magma", n_colors=len(df)), edgecolor="#3F0D12", linewidth=0.5)
    ax.set_title("Top Authors by Total Engagement")
    ax.set_xlabel("Total Engagement")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    annotate_bars(ax)
    save_figure(fig, "top_authors_by_total_engagement_barh")


def chart_clusters(top_clusters_df: pd.DataFrame) -> None:
    df = top_clusters_df.head(20).copy().sort_values("rows", ascending=True)
    labels = df["cluster_id"].astype(str) + " | " + df["theme_name"].str.slice(0, 35)
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.barh(labels, df["rows"], color=sns.color_palette("cubehelix", n_colors=len(df)), edgecolor="#1F2937", linewidth=0.4)
    ax.set_title("Largest Cluster-Theme Groups")
    ax.set_xlabel("Posts")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    annotate_bars(ax)
    save_figure(fig, "top_clusters_barh")


def chart_theme_author_summary(theme_author_df: pd.DataFrame) -> None:
    df = theme_author_df.copy()
    df["posts_per_author"] = df["rows"] / df["distinct_authors"]
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(
        df["distinct_authors"],
        df["rows"],
        s=np.clip(df["avg_likes"] * 20, 80, 700),
        c=df["avg_reposts"],
        cmap="viridis",
        alpha=0.9,
        edgecolor="white",
        linewidth=1.0,
    )
    for _, row in df.iterrows():
        ax.text(row["distinct_authors"] * 1.01, row["rows"] * 1.005, row["theme_name"][:26], fontsize=9)
    ax.set_title("Theme Scale: Authors vs Posts")
    ax.set_xlabel("Distinct Authors")
    ax.set_ylabel("Posts")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Average Reposts")
    save_figure(fig, "theme_authors_vs_posts_scatter")


def chart_theme_engagement_dot(theme_author_df: pd.DataFrame) -> None:
    df = theme_author_df.sort_values("avg_likes", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hlines(df["theme_name"], xmin=0, xmax=df["avg_likes"], color="#CBD5E1", linewidth=2)
    ax.scatter(df["avg_likes"], df["theme_name"], s=np.clip(df["avg_reposts"] * 50, 60, 500), color="#EA580C", alpha=0.9)
    ax.set_title("Average Likes by Theme")
    ax.set_xlabel("Average Likes per Post")
    ax.set_ylabel("")
    for x, y in zip(df["avg_likes"], df["theme_name"]):
        ax.text(x + 0.1, y, f"{x:.1f}", va="center", fontsize=9)
    save_figure(fig, "theme_avg_likes_dotplot")


def chart_author_account_age(author_age_df: pd.DataFrame) -> None:
    df = author_age_df.copy()
    df["author_created_month"] = parse_mixed_ts(df["author_created_month"])
    df = df.dropna(subset=["author_created_month"]).sort_values("author_created_month")
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(df["author_created_month"], df["posts"], width=25, color="#6366F1", edgecolor="#312E81", linewidth=0.4)
    ax.set_title("Posting Volume by Author Account Creation Month")
    ax.set_xlabel("")
    ax.set_ylabel("Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_figure(fig, "author_account_age_distribution")


def chart_content_shape(content_df: pd.DataFrame, total_rows: int) -> None:
    row = content_df.iloc[0]
    categories = [
        "External links",
        "Image embeds",
        "Media images",
        "Video embeds",
        "Hashtagged posts",
    ]
    values = [
        row["external_links"],
        row["image_embeds"],
        row["media_image_embeds"],
        row["video_embeds"],
        row["hashtagged_posts"],
    ]
    pct = [100 * v / total_rows for v in values]
    df = pd.DataFrame({"category": categories, "value": values, "pct": pct}).sort_values("value", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(df["category"], df["pct"], color=sns.color_palette("PuBuGn", n_colors=len(df)), edgecolor="#134E4A", linewidth=0.5)
    ax.set_title("Content Features as Share of Posts")
    ax.set_xlabel("Share of Posts (%)")
    ax.set_ylabel("")
    annotate_bars(ax, pct=True)
    save_figure(fig, "content_shape_share_barh")


def chart_engagement_zero_rates(engagement_df: pd.DataFrame, total_rows: int) -> None:
    row = engagement_df.iloc[0]
    categories = ["Zero likes", "Zero replies", "Zero reposts", "Zero quotes"]
    values = [row["zero_likes"], row["zero_replies"], row["zero_reposts"], row["zero_quotes"]]
    pct = [100 * v / total_rows for v in values]
    df = pd.DataFrame({"category": categories, "pct": pct}).sort_values("pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(df["category"], df["pct"], color=sns.color_palette("rocket", n_colors=len(df)), edgecolor="#3F0D12", linewidth=0.5)
    ax.set_title("Share of Posts With Zero Engagement by Type")
    ax.set_xlabel("Share of Posts (%)")
    ax.set_ylabel("")
    annotate_bars(ax, pct=True)
    save_figure(fig, "engagement_zero_rates_barh")


def chart_engagement_quantiles(engagement_df: pd.DataFrame) -> None:
    row = engagement_df.iloc[0]
    like_quantiles = parse_quantiles(row["like_quantiles"])
    reply_quantiles = parse_quantiles(row["reply_quantiles"])
    repost_quantiles = parse_quantiles(row["repost_quantiles"])
    quote_quantiles = parse_quantiles(row["quote_quantiles"])
    data = pd.DataFrame(
        {
            "metric": ["Likes", "Replies", "Reposts", "Quotes"],
            "p50": [like_quantiles[0], reply_quantiles[0], repost_quantiles[0], quote_quantiles[0]],
            "p90": [like_quantiles[1], reply_quantiles[1], repost_quantiles[1], quote_quantiles[1]],
            "p99": [like_quantiles[2], reply_quantiles[2], repost_quantiles[2], quote_quantiles[2]],
        }
    )
    melted = data.melt(id_vars="metric", var_name="quantile", value_name="value")
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=melted, x="metric", y="value", hue="quantile", palette="Set1", ax=ax)
    ax.set_title("Engagement Quantiles by Interaction Type")
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    save_figure(fig, "engagement_quantiles_grouped_bar")


def chart_partition_distribution(partition_df: pd.DataFrame) -> None:
    df = partition_df.sort_values("rows", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df["data_partition"], df["rows"], color=sns.color_palette("Set2", n_colors=len(df)), edgecolor="#1F2937", linewidth=0.5)
    ax.set_title("Rows by Data Partition")
    ax.set_xlabel("Rows")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    annotate_bars(ax)
    save_figure(fig, "partition_distribution_barh")


def chart_file_row_counts(file_counts_df: pd.DataFrame) -> None:
    df = file_counts_df.copy()
    df["short_file"] = ["Part 1", "Part 2", "Part 3", "Part 4"]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=df, x="short_file", y="rows", palette="Blues", ax=ax)
    ax.set_title("Rows per Source File")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2, p.get_height() + df["rows"].max() * 0.015, f"{int(p.get_height()):,}", ha="center", fontsize=10)
    save_figure(fig, "file_row_counts_bar")


def chart_file_timeline(per_file_df: pd.DataFrame) -> None:
    df = per_file_df.copy()
    df["min_created"] = parse_mixed_ts(df["min_created"])
    df["max_created"] = parse_mixed_ts(df["max_created"])
    df = df.sort_values("min_created")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = sns.color_palette("coolwarm", n_colors=len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        ax.barh(
            row["file"].replace("_merged_with_microtopics_input_", "\n"),
            (row["max_created"] - row["min_created"]).days + 1,
            left=row["min_created"],
            height=0.6,
            color=colors[i],
            edgecolor="#1F2937",
            linewidth=0.5,
        )
    ax.set_title("Record Creation Coverage by File")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    save_figure(fig, "file_creation_timeline")


def chart_theme_cumulative(theme_monthly_df: pd.DataFrame) -> None:
    df = theme_monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    fig, ax = plt.subplots(figsize=(15, 7))
    palette = sns.color_palette("Dark2", n_colors=df["theme_name"].nunique())
    for color, (theme, sub) in zip(palette, df.groupby("theme_name")):
        sub = sub.sort_values("month").copy()
        sub["cum_rows"] = sub["rows"].cumsum()
        ax.plot(sub["month"], sub["cum_rows"], label=theme, linewidth=2.2, color=color)
    ax.set_title("Cumulative Growth of Top 8 Themes")
    ax.set_xlabel("")
    ax.set_ylabel("Cumulative Posts")
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    save_figure(fig, "top8_theme_cumulative_lines")


def chart_theme_slope(theme_monthly_df: pd.DataFrame) -> None:
    df = theme_monthly_df.copy()
    df["month"] = parse_mixed_ts(df["month"])
    latest = df["month"].max()
    prev = sorted(df["month"].unique())[-2]
    sub = df[df["month"].isin([prev, latest])].pivot(index="theme_name", columns="month", values="rows").fillna(0)
    sub["change"] = sub[latest] - sub[prev]
    sub = sub.sort_values("change")
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ["#DC2626" if x < 0 else "#16A34A" for x in sub["change"]]
    ax.hlines(sub.index, sub[prev], sub[latest], color="#CBD5E1", linewidth=2)
    ax.scatter(sub[prev], sub.index, color="#64748B", s=80, label=prev.strftime("%Y-%m"))
    ax.scatter(sub[latest], sub.index, color=colors, s=110, label=latest.strftime("%Y-%m"))
    ax.set_title(f"Month-over-Month Change in Top 8 Themes: {prev.strftime('%Y-%m')} to {latest.strftime('%Y-%m')}")
    ax.set_xlabel("Posts")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(fmt_int))
    ax.legend(loc="lower right")
    save_figure(fig, "top8_theme_latest_slope")


def chart_overview_kpi(overview_df: pd.DataFrame) -> None:
    row = overview_df.iloc[0]
    metrics = pd.DataFrame(
        {
            "metric": ["Rows", "Authors", "Clusters", "Themes"],
            "value": [
                row["total_rows"],
                row["distinct_authors"],
                row["distinct_clusters"],
                row["distinct_themes"],
            ],
        }
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    xs = np.linspace(0.12, 0.88, len(metrics))
    colors = ["#0F766E", "#2563EB", "#EA580C", "#7C3AED"]
    for x, color, (_, r) in zip(xs, colors, metrics.iterrows()):
        ax.text(x, 0.62, f"{int(r['value']):,}", ha="center", va="center", fontsize=24, fontweight="bold", color=color)
        ax.text(x, 0.42, r["metric"], ha="center", va="center", fontsize=13, color="#334155")
    ax.set_title("Dataset Overview", pad=20)
    save_figure(fig, "dataset_overview_kpi")


def query_additional_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect()
    theme_year = con.execute(
        f"""
        SELECT
          year(try_cast("record.created_at" AS TIMESTAMPTZ)) AS year,
          "Theme_Name" AS theme_name,
          count(*) AS rows
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
        GROUP BY 1, 2
        HAVING year IS NOT NULL
        ORDER BY 1, 3 DESC
        """
    ).df()
    top_authors_theme = con.execute(
        f"""
        WITH top_authors AS (
          SELECT "author.handle" AS author_handle
          FROM read_parquet('{PARQUET_GLOB}')
          GROUP BY 1
          ORDER BY count(*) DESC
          LIMIT 12
        )
        SELECT
          "author.handle" AS author_handle,
          "Theme_Name" AS theme_name,
          count(*) AS rows
        FROM read_parquet('{PARQUET_GLOB}')
        WHERE "author.handle" IN (SELECT author_handle FROM top_authors)
        GROUP BY 1, 2
        ORDER BY 1, 3 DESC
        """
    ).df()
    return theme_year, top_authors_theme


def chart_theme_year_heatmap(theme_year_df: pd.DataFrame) -> None:
    df = theme_year_df.copy()
    pivot = df.pivot(index="theme_name", columns="year", values="rows").fillna(0)
    pivot = pivot.loc[df.groupby("theme_name")["rows"].sum().sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot, cmap="YlOrRd", linewidths=0.2, linecolor="white", annot=True, fmt=".0f", cbar_kws={"format": FuncFormatter(fmt_int)}, ax=ax)
    ax.set_title("Theme Distribution by Year")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save_figure(fig, "theme_year_heatmap")


def chart_top_author_theme_mix(top_authors_theme_df: pd.DataFrame) -> None:
    df = top_authors_theme_df.copy()
    pivot = df.pivot(index="author_handle", columns="theme_name", values="rows").fillna(0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    share = pivot.div(pivot.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(15, 7))
    colors = sns.color_palette("tab20", n_colors=share.shape[1])
    left = np.zeros(len(share))
    for color, col in zip(colors, share.columns):
        ax.barh(share.index, share[col], left=left, color=color, label=col)
        left += share[col].to_numpy()
    ax.set_title("Theme Mix for Top Posting Authors")
    ax.set_xlabel("Share of Each Author's Posts")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), title="")
    save_figure(fig, "top_authors_theme_mix_stacked")


def main() -> None:
    ensure_dirs()
    set_theme()

    overview_df = load_csv("overview_metrics.csv")
    file_counts_df = load_csv("file_row_counts.csv")
    partition_df = load_csv("partition_distribution.csv")
    per_file_df = load_csv("per_file_summary.csv")
    theme_df = load_csv("theme_distribution.csv")
    monthly_df = load_csv("monthly_trends.csv")
    theme_monthly_df = load_csv("theme_monthly_top8.csv")
    top_lang_df = load_csv("top_languages.csv")
    content_df = load_csv("content_shape_metrics.csv")
    engagement_df = load_csv("engagement_distribution.csv")
    authors_posts_df = load_csv("top_authors_by_posts.csv")
    authors_eng_df = load_csv("top_authors_by_total_engagement.csv")
    top_clusters_df = load_csv("top_clusters.csv")
    theme_author_df = load_csv("theme_author_summary.csv")
    author_age_df = load_csv("author_account_age.csv")

    total_rows = int(overview_df.loc[0, "total_rows"])
    theme_year_df, top_authors_theme_df = query_additional_tables()

    chart_overview_kpi(overview_df)
    chart_file_row_counts(file_counts_df)
    chart_partition_distribution(partition_df)
    chart_file_timeline(per_file_df)
    chart_theme_distribution(theme_df)
    chart_theme_share(theme_df)
    chart_theme_confidence(theme_df)
    chart_monthly_volume(monthly_df)
    chart_monthly_volume_log(monthly_df)
    chart_monthly_heatmap(monthly_df)
    chart_top8_theme_lines(theme_monthly_df)
    chart_top8_theme_stacked_area(theme_monthly_df)
    chart_top8_theme_share_area(theme_monthly_df)
    chart_top8_theme_heatmap(theme_monthly_df)
    chart_theme_cumulative(theme_monthly_df)
    chart_theme_slope(theme_monthly_df)
    chart_theme_year_heatmap(theme_year_df)
    chart_top_languages(top_lang_df)
    chart_top_authors_posts(authors_posts_df)
    chart_top_authors_engagement(authors_eng_df)
    chart_top_author_theme_mix(top_authors_theme_df)
    chart_clusters(top_clusters_df)
    chart_theme_author_summary(theme_author_df)
    chart_theme_engagement_dot(theme_author_df)
    chart_author_account_age(author_age_df)
    chart_content_shape(content_df, total_rows)
    chart_engagement_zero_rates(engagement_df, total_rows)
    chart_engagement_quantiles(engagement_df)

    figure_index = sorted(p.name for p in FIG_DIR.iterdir() if p.is_file())
    (FIG_DIR / "figure_manifest.txt").write_text("\n".join(figure_index) + "\n", encoding="utf-8")
    print(f"Saved {len(figure_index)} figure files to {FIG_DIR}")


if __name__ == "__main__":
    main()
