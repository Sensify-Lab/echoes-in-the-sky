from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from statsmodels.tsa.stattools import grangercausalitytests

BASE_DIR = Path("local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs")
STATIONARITY_DIR = BASE_DIR / "stationarity_analysis"
OUT_DIR = BASE_DIR / "stationary_granger_analysis_corrected_May20"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "publication_figures"

INPUT_CSV = STATIONARITY_DIR / "tables" / "granger_stationary_input_recommended.csv"
REC_CSV = STATIONARITY_DIR / "tables" / "stationarity_recommendations.csv"
MAX_LAG = 14

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def run_granger(df: pd.DataFrame, direction: str) -> pd.DataFrame:
    rows = []
    tested_pairs = []
    skipped_pairs = []

    for (bluesky_theme, eo_theme), sub in df.groupby(["bluesky_theme", "eo_theme"], sort=True):
        meta = sub.iloc[0]
        pair_id = f"{bluesky_theme} | {eo_theme}"
        sub = sub.sort_values("day").copy()

        if meta["pair_granger_ready"] == "needs_manual_review":
            skipped_pairs.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bluesky_theme,
                    "eo_theme": eo_theme,
                    "mapping_confidence": meta["mapping_confidence"],
                    "pair_granger_ready": meta["pair_granger_ready"],
                    "eo_transform_used": meta["eo_count_transform_used"],
                    "bluesky_transform_used": meta["bluesky_posts_transform_used"],
                    "reason": "stationarity_manual_review",
                }
            )
            continue

        if direction == "bluesky_to_eo":
            sub2 = sub[["eo_count_stationary", "bluesky_posts_stationary"]].dropna().copy()
            tested_var = "bluesky_posts_stationary"
            target_var = "eo_count_stationary"
        else:
            sub2 = sub[["bluesky_posts_stationary", "eo_count_stationary"]].dropna().copy()
            tested_var = "eo_count_stationary"
            target_var = "bluesky_posts_stationary"

        if len(sub2) <= MAX_LAG + 5:
            skipped_pairs.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bluesky_theme,
                    "eo_theme": eo_theme,
                    "mapping_confidence": meta["mapping_confidence"],
                    "pair_granger_ready": meta["pair_granger_ready"],
                    "eo_transform_used": meta["eo_count_transform_used"],
                    "bluesky_transform_used": meta["bluesky_posts_transform_used"],
                    "reason": "insufficient_post_transform_rows",
                }
            )
            continue

        try:
            tests = grangercausalitytests(sub2, maxlag=MAX_LAG, verbose=False)
        except Exception as exc:
            skipped_pairs.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bluesky_theme,
                    "eo_theme": eo_theme,
                    "mapping_confidence": meta["mapping_confidence"],
                    "pair_granger_ready": meta["pair_granger_ready"],
                    "eo_transform_used": meta["eo_count_transform_used"],
                    "bluesky_transform_used": meta["bluesky_posts_transform_used"],
                    "reason": f"granger_error:{type(exc).__name__}",
                }
            )
            continue

        tested_pairs.append(
            {
                "pair_id": pair_id,
                "bluesky_theme": bluesky_theme,
                "eo_theme": eo_theme,
                "mapping_confidence": meta["mapping_confidence"],
                "pair_granger_ready": meta["pair_granger_ready"],
                "eo_transform_used": meta["eo_count_transform_used"],
                "bluesky_transform_used": meta["bluesky_posts_transform_used"],
                "nobs_after_transform": int(len(sub2)),
                "direction": direction,
            }
        )

        for lag, payload in tests.items():
            ftest = payload[0]["ssr_ftest"]
            rows.append(
                {
                    "direction": direction,
                    "pair_id": pair_id,
                    "bluesky_theme": bluesky_theme,
                    "eo_theme": eo_theme,
                    "mapping_confidence": meta["mapping_confidence"],
                    "pair_granger_ready": meta["pair_granger_ready"],
                    "eo_transform_used": meta["eo_count_transform_used"],
                    "bluesky_transform_used": meta["bluesky_posts_transform_used"],
                    "tested_var": tested_var,
                    "target_var": target_var,
                    "lag_days": int(lag),
                    "f_statistic": float(ftest[0]),
                    "p_value": float(ftest[1]),
                    "df_denom": float(ftest[2]),
                    "df_num": float(ftest[3]),
                    "significant_005": bool(ftest[1] < 0.05),
                    "nobs_after_transform": int(len(sub2)),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(tested_pairs), pd.DataFrame(skipped_pairs)

def plot_fstats(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return

    order = (
        df.groupby("pair_id")["f_statistic"]
        .max()
        .sort_values(ascending=True)
        .index
    )
    plot_df = df.copy()
    plot_df["pair_id"] = pd.Categorical(plot_df["pair_id"], categories=list(order), ordered=True)
    plot_df = plot_df.sort_values(["pair_id", "lag_days"])

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(18, max(7, 0.65 * len(order))))

    sns.boxplot(
        data=plot_df,
        y="pair_id",
        x="f_statistic",
        color="white",
        width=0.6,
        fliersize=0,
        linewidth=1.2,
        ax=ax,
    )

    sig = plot_df["significant_005"]
    ax.scatter(
        plot_df.loc[sig, "f_statistic"],
        plot_df.loc[sig, "pair_id"],
        color="#2b83ba",
        s=26,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        plot_df.loc[~sig, "f_statistic"],
        plot_df.loc[~sig, "pair_id"],
        color="#d7191c",
        s=20,
        alpha=0.8,
        zorder=3,
    )

    y_positions = {pair: idx for idx, pair in enumerate(order)}
    lag_offsets = {
        1: -16, 2: -10, 3: -4, 4: 2, 5: 8, 6: 14, 7: -13,
        8: -7, 9: -1, 10: 5, 11: 11, 12: 17, 13: -19, 14: -5,
    }
    x_offsets = {
        1: 8, 2: 10, 3: 12, 4: 8, 5: 10, 6: 12, 7: 8,
        8: 10, 9: 12, 10: 8, 11: 10, 12: 12, 13: 8, 14: 10,
    }
    for _, row in plot_df.loc[sig].iterrows():
        ax.annotate(
            f"D{int(row['lag_days'])}",
            xy=(row["f_statistic"], y_positions[row["pair_id"]]),
            xytext=(x_offsets.get(int(row["lag_days"]), 8), lag_offsets.get(int(row["lag_days"]), 0)),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#444444",
            alpha=0.95,
        )

    ax.set_title(title, loc="left", fontsize=18)
    ax.set_xlabel("F-statistic")
    ax.set_ylabel("Theme pair")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="p-value < 0.05", markerfacecolor="#2b83ba", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="p-value >= 0.05", markerfacecolor="#d7191c", markersize=8),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_min_pvalue_bar(df: pd.DataFrame, out_path: Path) -> None:
    if df.empty:
        return
    best = (
        df.groupby("pair_id", as_index=False)["p_value"]
        .min()
        .sort_values("p_value", ascending=True)
    )
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(16, max(7, 0.6 * len(best))))
    sns.barplot(data=best, y="pair_id", x="p_value", color="#4c78a8", ax=ax)
    ax.axvline(0.05, color="#7f0000", linestyle="--", linewidth=1.8)
    ax.set_title("Minimum Granger p-value by Pair", loc="left", fontsize=18)
    ax.set_xlabel("Minimum p-value across D1-D14")
    ax.set_ylabel("Theme pair")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    df = pd.read_csv(INPUT_CSV)
    df["day"] = pd.to_datetime(df["day"])
    recommendations = pd.read_csv(REC_CSV)

    b2e_rows, b2e_tested, b2e_skipped = run_granger(df, "bluesky_to_eo")
    e2b_rows, e2b_tested, e2b_skipped = run_granger(df, "eo_to_bluesky")

    b2e_rows.to_csv(TABLE_DIR / "bluesky_to_eo_granger_fstats_stationary.csv", index=False)
    e2b_rows.to_csv(TABLE_DIR / "eo_to_bluesky_granger_fstats_stationary.csv", index=False)
    b2e_tested.to_csv(TABLE_DIR / "bluesky_to_eo_tested_pairs_stationary.csv", index=False)
    e2b_tested.to_csv(TABLE_DIR / "eo_to_bluesky_tested_pairs_stationary.csv", index=False)
    pd.concat([b2e_skipped.assign(direction="bluesky_to_eo"), e2b_skipped.assign(direction="eo_to_bluesky")], ignore_index=True).to_csv(
        TABLE_DIR / "granger_skipped_pairs_stationary.csv", index=False
    )

    plot_fstats(
        b2e_rows,
        "Stationarity-Corrected Granger F-statistics: Bluesky -> EO",
        FIG_DIR / "bluesky_to_eo_granger_fstats_stationary",
    )
    plot_fstats(
        e2b_rows,
        "Stationarity-Corrected Granger F-statistics: EO -> Bluesky",
        FIG_DIR / "eo_to_bluesky_granger_fstats_stationary",
    )
    plot_min_pvalue_bar(b2e_rows, FIG_DIR / "bluesky_to_eo_min_pvalue_stationary")
    plot_min_pvalue_bar(e2b_rows, FIG_DIR / "eo_to_bluesky_min_pvalue_stationary")

    print(f"b2e\t{TABLE_DIR / 'bluesky_to_eo_granger_fstats_stationary.csv'}")
    print(f"e2b\t{TABLE_DIR / 'eo_to_bluesky_granger_fstats_stationary.csv'}")
    print(f"skipped\t{TABLE_DIR / 'granger_skipped_pairs_stationary.csv'}")
    print(f"figures\t{FIG_DIR}")

if __name__ == "__main__":
    main()
