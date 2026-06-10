from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


BASE_DIR = Path(__file__).resolve().parent
TABLE_DIR = BASE_DIR / "tables"
OUTPUT_DIR = BASE_DIR / "publication_figures_tlabels"

THEME_LABELS = [
    "T1 Executive Power",
    "T2 Partisan Conflict",
    "T3 National Security",
    "T4 Authoritarianism",
    "T5 Media Narratives",
    "T6 Legal Rights",
    "T7 Economic Trade",
    "T8 Technology",
    "T9 Public Lands",
    "T10 Education",
    "T11 Public Health",
    "T12 Immigration",
    "T13 Symbolic Symbolism",
    "T14 Race Rights",
]


def build_count_heatmap(counts: pd.DataFrame) -> None:
    theme_order = (
        counts.groupby("theme", as_index=False)["post_count"]
        .sum()
        .sort_values("post_count", ascending=False)["theme"]
        .tolist()
    )

    totals = counts.groupby(["theme", "month"], as_index=False)["post_count"].sum()
    theme_totals = (
        totals.groupby("theme", as_index=False)["post_count"]
        .sum()
        .rename(columns={"post_count": "theme_total"})
    )
    totals = totals.merge(theme_totals, on="theme", how="left")
    totals["theme_month_pct"] = totals["post_count"] / totals["theme_total"] * 100.0

    count_pivot = (
        totals.pivot(index="theme", columns="month", values="post_count")
        .reindex(theme_order)
    )
    pct_pivot = (
        totals.pivot(index="theme", columns="month", values="theme_month_pct")
        .reindex(theme_order)
    )

    fig, ax = plt.subplots(figsize=(16, max(7, len(theme_order) * 0.48)))

    im = ax.imshow(
        count_pivot.values,
        aspect="auto",
        cmap="YlGnBu",
        norm=LogNorm(
            vmin=max(np.nanmin(count_pivot.values[count_pivot.values > 0]), 1),
            vmax=np.nanmax(count_pivot.values),
        ),
    )

    months = list(count_pivot.columns)
    ax.set_yticks(range(len(theme_order)))
    ax.set_yticklabels(THEME_LABELS, fontsize=16, fontweight="bold")
    ax.set_xticks(range(len(months)))
    ax.set_xticklabels(
        [pd.to_datetime(m).strftime("%Y-%m") for m in months],
        rotation=45,
        ha="right",
        fontsize=11,
        fontweight="bold",
    )

    max_val = np.nanmax(count_pivot.values)
    for i in range(count_pivot.shape[0]):
        for j in range(count_pivot.shape[1]):
            val = count_pivot.iat[i, j]
            pct = pct_pivot.iat[i, j]
            if pd.isna(val):
                continue
            color = "white" if val >= max_val * 0.25 else "black"
            ax.text(
                j,
                i,
                f"{pct:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, pad=0.015)
    cbar.set_label("Monthly post count (log scale)", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=16)

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "theme_month_count_heatmap_tlabels.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "theme_month_count_heatmap_tlabels.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)




def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    counts = pd.read_csv(TABLE_DIR / "theme_month_counts_from_2023_06.csv", parse_dates=["month"])
    shares = pd.read_csv(TABLE_DIR / "theme_month_shares_from_2023_06.csv", parse_dates=["month"])
    build_count_heatmap(counts)
    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
