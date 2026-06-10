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

SRC_DIR = Path(
    "local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs/"
    "stationary_granger_analysis_corrected_May20"
)
OUT_DIR = Path(
    "local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs/"
    "stationary_granger_analysis_significant_labels_reversed"
)
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "publication_figures"

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load(name: str) -> pd.DataFrame:
    return pd.read_csv(SRC_DIR / "tables" / name)

def plot_fstats(df: pd.DataFrame, title: str, out_path: Path) -> None:
    if df.empty:
        return

    # Reverse the prior ranking so the strongest pair appears at the top.
    order = (
        df.groupby("pair_id")["f_statistic"]
        .max()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    plot_df = df.copy()
    plot_df["pair_id"] = pd.Categorical(plot_df["pair_id"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["pair_id", "lag_days"])

    # Keep only one D label per pair: the most significant lag, tie-broken by larger F-statistic.
    sig_df = plot_df.loc[plot_df["significant_005"]].copy()
    if not sig_df.empty:
        label_df = (
            sig_df.sort_values(["pair_id", "p_value", "f_statistic"], ascending=[True, True, False])
            .groupby("pair_id", as_index=False, observed=True)
            .first()
        )
        label_df = label_df.dropna(subset=["lag_days", "f_statistic"])
    else:
        label_df = sig_df

    sns.set_theme(style="white", context="talk")
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
        s=28,
        alpha=0.92,
        zorder=3,
    )
    ax.scatter(
        plot_df.loc[~sig, "f_statistic"],
        plot_df.loc[~sig, "pair_id"],
        color="#d7191c",
        s=20,
        alpha=0.82,
        zorder=3,
    )

    y_positions = {pair: idx for idx, pair in enumerate(order)}
    for i, (_, row) in enumerate(label_df.iterrows()):
        yoff = -10 if i % 2 == 0 else 10
        ax.annotate(
            f"D{int(row['lag_days'])}",
            xy=(row["f_statistic"], y_positions[row["pair_id"]]),
            xytext=(10, yoff),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            color="#333333",
            fontweight="bold",
            alpha=0.98,
        )

    ax.set_title(title, loc="left", fontsize=18)
    ax.set_xlabel("F-statistic")
    ax.set_ylabel("Theme pair")
    ax.grid(False)

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label="p-value < 0.05", markerfacecolor="#2b83ba", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="p-value >= 0.05", markerfacecolor="#d7191c", markersize=9),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=16, title_fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    b2e = load("bluesky_to_eo_granger_fstats_stationary.csv")
    e2b = load("eo_to_bluesky_granger_fstats_stationary.csv")

    # Copy source tables for provenance.
    b2e.to_csv(TABLE_DIR / "bluesky_to_eo_granger_fstats_stationary.csv", index=False)
    e2b.to_csv(TABLE_DIR / "eo_to_bluesky_granger_fstats_stationary.csv", index=False)

    plot_fstats(
        b2e,
        "Stationarity-Corrected Granger F-statistics: Bluesky -> EO",
        FIG_DIR / "bluesky_to_eo_granger_fstats_stationary",
    )
    plot_fstats(
        e2b,
        "Stationarity-Corrected Granger F-statistics: EO -> Bluesky",
        FIG_DIR / "eo_to_bluesky_granger_fstats_stationary",
    )

    print(f"figures\t{FIG_DIR}")
    print(f"tables\t{TABLE_DIR}")

if __name__ == "__main__":
    main()
