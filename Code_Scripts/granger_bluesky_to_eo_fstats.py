from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import grangercausalitytests


BASE_DIR = Path("local_data/03_Github_data/07_EO_effect_bluesky_2025_01_20_to_2026_02_01")
SOURCE_TABLE_DIR = BASE_DIR / "tables"
OUTPUT_DIR = BASE_DIR / "bluesky_to_eo_granger_fstats"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "publication_figures"

WINDOW_START = "2025-01-20"
WINDOW_END = "2026-02-01"
MAX_LAG_DAYS = 14


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def compute_detailed_granger() -> pd.DataFrame:
    eo_daily = pd.read_csv(SOURCE_TABLE_DIR / "eo_theme_daily.csv")
    bs_daily = pd.read_csv(SOURCE_TABLE_DIR / "bluesky_theme_daily.csv")
    eo_daily["day"] = pd.to_datetime(eo_daily["day"])
    bs_daily["day"] = pd.to_datetime(bs_daily["day"])
    themes = sorted(set(eo_daily["bluesky_theme"].dropna()) & set(bs_daily["bluesky_theme"].dropna()))

    rows = []
    input_rows = []
    eligible_input_rows = []
    for theme in themes:
        eo_sub = eo_daily[eo_daily["bluesky_theme"] == theme].groupby("day", as_index=True)["eo_count"].sum()
        bs_sub = bs_daily[bs_daily["bluesky_theme"] == theme].groupby("day", as_index=True)["bluesky_posts"].sum()
        idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
        eo_day = eo_sub.reindex(idx, fill_value=0).astype(float)
        bs_day = bs_sub.reindex(idx, fill_value=0).astype(float)
        df = pd.DataFrame({"eo": eo_day, "bluesky": bs_day}).reset_index(drop=True)

        df.to_csv("df_checkingpoint.csv", index=False)

        input_df = pd.DataFrame(
            {
                "day": idx,
                "theme": theme,
                "eo_count": eo_day.values,
                "bluesky_posts": bs_day.values,
            }
        )
        input_rows.append(input_df)
        if len(df) < (MAX_LAG_DAYS * 3) or df["eo"].std() == 0 or df["bluesky"].sum() == 0:
            continue
        eligible_input_rows.append(input_df)
        try:
            results = grangercausalitytests(df[["eo", "bluesky"]], maxlag=MAX_LAG_DAYS, verbose=False)
        except Exception:
            continue
        for lag, res in results.items():
            f_stat, p_value, df_denom, df_num = res[0]["ssr_ftest"]
            rows.append(
                {
                    "relationship": f"{theme}  Bluesky → EO",
                    "theme": theme,
                    "lag_days": lag,
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "df_denom": float(df_denom),
                    "df_num": float(df_num),
                    "significant_005": bool(p_value < 0.05),
                }
            )

    if input_rows:
        pd.concat(input_rows, ignore_index=True).to_csv(
            TABLE_DIR / "bluesky_to_eo_granger_input_daily.csv",
            index=False,
        )
    if eligible_input_rows:
        pd.concat(eligible_input_rows, ignore_index=True).to_csv(
            TABLE_DIR / "bluesky_to_eo_granger_input_daily_eligible.csv",
            index=False,
        )
    out = pd.DataFrame(rows).sort_values(["theme", "lag_days"])
    out.to_csv(TABLE_DIR / "bluesky_to_eo_granger_fstats_detailed.csv", index=False)
    return out


def plot_fstats(df: pd.DataFrame) -> None:
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

    order = (
        df.groupby("relationship")["f_statistic"]
        .median()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    df = df.copy()
    df["relationship"] = pd.Categorical(df["relationship"], categories=order, ordered=True)
    df = df.sort_values("relationship")

    fig, ax = plt.subplots(figsize=(18, max(8, 0.7 * len(order))))
    sns.boxplot(
        data=df,
        y="relationship",
        x="f_statistic",
        order=order,
        color="white",
        width=0.55,
        fliersize=0,
        linewidth=1.1,
        ax=ax,
    )

    y_positions = {label: i for i, label in enumerate(order)}
    rng = np.random.default_rng(42)
    for _, row in df.iterrows():
        y = y_positions[row["relationship"]] + rng.uniform(-0.12, 0.12)
        color = "#3182BD" if row["significant_005"] else "#E15759"
        ax.scatter(
            row["f_statistic"],
            y,
            s=42,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
        )
        ax.text(
            row["f_statistic"] + 0.03,
            y,
            f"D{int(row['lag_days'])}",
            fontsize=8,
            va="center",
            color=color,
        )

    ax.set_title("Bluesky → EO Granger Causality F-statistics (Daily Lags)")
    ax.set_xlabel("F-statistic")
    ax.set_ylabel("Mapped Theme Relationship")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3182BD", markeredgecolor="white", markersize=9, label="p-value < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E15759", markeredgecolor="white", markersize=9, label="p-value ≥ 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    stem = "bluesky_to_eo_granger_fstats"
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    df = compute_detailed_granger()
    plot_fstats(df)
    print(f"input\t{TABLE_DIR / 'bluesky_to_eo_granger_input_daily.csv'}")
    print(f"eligible_input\t{TABLE_DIR / 'bluesky_to_eo_granger_input_daily_eligible.csv'}")
    print(f"table\t{TABLE_DIR / 'bluesky_to_eo_granger_fstats_detailed.csv'}")
    print(f"figure\t{FIG_DIR / 'bluesky_to_eo_granger_fstats.png'}")


if __name__ == "__main__":
    main()
