from __future__ import annotations

import csv
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
OUTPUT_DIR = BASE_DIR / "eo19_to_bluesky_granger"
TABLE_DIR = OUTPUT_DIR / "tables"
FIG_DIR = OUTPUT_DIR / "publication_figures"
CROSSWALK_CSV = OUTPUT_DIR / "eo19_to_bluesky_theme_crosswalk.csv"

WINDOW_START = "2025-01-20"
WINDOW_END = "2026-02-01"
MAX_LAG_DAYS = 14

CROSSWALK_ROWS = [
    ("Executive Coordination and Administrative Task Forces", "Executive Power, Patronage, and Administrative Control", "high", "Direct executive-administration match."),
    ("Public Order, Crime, and Border Enforcement", "Immigration, Borders, and Sovereignty", "medium", "Closest match on border enforcement and sovereignty."),
    ("Foreign Policy and External Relations", "National Security, Military, and Geopolitical Conflict", "high", "Direct foreign-policy and geopolitical match."),
    ("National Security, Defense, and Strategic Resilience", "National Security, Military, and Geopolitical Conflict", "high", "Direct national-security match."),
    ("Trade, Tariffs, and Export Promotion", "Economic Policy, Taxation, and Distributional Conflict", "high", "Representative economic-policy anchor."),
    ("Domestic Investment and Market Competition", "Economic Policy, Taxation, and Distributional Conflict", "medium", "Domestic market and investment policy anchor."),
    ("Civil Rights, Equality, and Gender Policy", "Race, Civil Rights, and Identity Hierarchies", "medium", "Closest civil-rights and equality match."),
    ("Health, Reproductive Care, and Medical Research", "Health, Public Health, and Bodily Politics", "high", "Direct health-policy match."),
    ("Energy Development and Resource Expansion", "Public Lands, Environment, and Climate Governance", "medium", "Resource extraction and environmental governance link."),
    ("Education and Higher-Ed Transparency", "Education, Universities, and Intellectual Control", "high", "Direct education-governance match."),
    ("Technology, Science, and Artificial Intelligence Development", "Technology, Platforms, and Digital Power", "high", "Direct technology governance match."),
    ("Workforce, Training, and Retirement Security", "Economic Policy, Taxation, and Distributional Conflict", "medium", "Labor and social provision within economic policy frame."),
    ("Cultural Heritage and National Pastime", "Aesthetics, Symbolism, and Performance of Power", "medium", "Closest symbolic politics match."),
    ("Environmental Stewardship and Beautification", "Public Lands, Environment, and Climate Governance", "high", "Direct environmental governance match."),
    ("Housing Affordability and Development Deregulation", "Economic Policy, Taxation, and Distributional Conflict", "medium", "Housing-development policy inside broader economic governance."),
    ("Infrastructure and Maritime Transportation Rules", "Economic Policy, Taxation, and Distributional Conflict", "low", "Exploratory placement in economic governance."),
    ("Public Lands, Parks, and Timber Production", "Public Lands, Environment, and Climate Governance", "high", "Direct land and resource governance match."),
    ("Water Supply and Disaster Response", "Public Lands, Environment, and Climate Governance", "low", "Exploratory placement in environment and resource governance."),
    ("Family Support and Child Wellbeing", "Health, Public Health, and Bodily Politics", "low", "Closest social-policy and bodily politics link."),
]


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def write_crosswalk() -> pd.DataFrame:
    with CROSSWALK_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eo_theme", "bluesky_theme", "mapping_confidence", "mapping_notes"])
        writer.writerows(CROSSWALK_ROWS)
    return pd.DataFrame(
        CROSSWALK_ROWS,
        columns=["eo_theme", "bluesky_theme", "mapping_confidence", "mapping_notes"],
    )


def load_sources() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bs_daily = pd.read_csv(SOURCE_TABLE_DIR / "bluesky_theme_daily.csv")
    eo_daily = pd.read_csv(SOURCE_TABLE_DIR / "eo_theme_daily.csv")
    crosswalk = pd.read_csv(CROSSWALK_CSV)
    bs_daily["day"] = pd.to_datetime(bs_daily["day"])
    eo_daily["day"] = pd.to_datetime(eo_daily["day"])
    return bs_daily, eo_daily, crosswalk


def build_mapped_input(bs_daily: pd.DataFrame, eo_daily: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    rows = []
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")

    for row in crosswalk.itertuples(index=False):
        eo_theme = row.eo_theme
        bs_theme = row.bluesky_theme
        eo_sub = (
            eo_daily[eo_daily["eo_theme"] == eo_theme]
            .groupby("day", as_index=True)["eo_count"]
            .sum()
            .reindex(idx, fill_value=0)
            .astype(float)
        )
        bs_sub = (
            bs_daily[bs_daily["bluesky_theme"] == bs_theme]
            .groupby("day", as_index=True)["bluesky_posts"]
            .sum()
            .reindex(idx, fill_value=0)
            .astype(float)
        )
        panel = pd.DataFrame(
            {
                "day": idx,
                "eo_theme": eo_theme,
                "bluesky_theme": bs_theme,
                "mapping_confidence": row.mapping_confidence,
                "mapping_notes": row.mapping_notes,
                "eo_count": eo_sub.values,
                "bluesky_posts": bs_sub.values,
            }
        )
        rows.append(panel)

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(TABLE_DIR / "eo19_to_bluesky_granger_input_daily.csv", index=False)
    crosswalk.to_csv(TABLE_DIR / "eo19_to_bluesky_theme_crosswalk_inventory.csv", index=False)
    return out


def granger_eligibility(panel: pd.DataFrame, direction: str) -> bool:
    if len(panel) < (MAX_LAG_DAYS * 3):
        return False
    if direction == "eo_to_bluesky":
        return panel["eo_count"].sum() > 0 and panel["bluesky_posts"].std() > 0
    return panel["eo_count"].std() > 0 and panel["bluesky_posts"].sum() > 0


def compute_granger(mapped_input: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eligible_panels = []

    for (eo_theme, bs_theme), sub in mapped_input.groupby(["eo_theme", "bluesky_theme"], sort=True):
        panel = sub.sort_values("day").reset_index(drop=True)
        if not granger_eligibility(panel, direction):
            continue
        eligible_panels.append(panel)
        test_df = panel[["bluesky_posts", "eo_count"]] if direction == "eo_to_bluesky" else panel[["eo_count", "bluesky_posts"]]
        try:
            results = grangercausalitytests(test_df, maxlag=MAX_LAG_DAYS, verbose=False)
        except Exception:
            continue

        for lag, res in results.items():
            f_stat, p_value, df_denom, df_num = res[0]["ssr_ftest"]
            rows.append(
                {
                    "eo_theme": eo_theme,
                    "bluesky_theme": bs_theme,
                    "mapping_confidence": panel["mapping_confidence"].iloc[0],
                    "relationship": f"{eo_theme}  |  {bs_theme}",
                    "lag_days": lag,
                    "f_statistic": float(f_stat),
                    "p_value": float(p_value),
                    "df_denom": float(df_denom),
                    "df_num": float(df_num),
                    "significant_005": bool(p_value < 0.05),
                }
            )

    detail_df = pd.DataFrame(rows).sort_values(["eo_theme", "lag_days"])
    eligible_df = pd.concat(eligible_panels, ignore_index=True) if eligible_panels else mapped_input.iloc[0:0].copy()

    if direction == "eo_to_bluesky":
        eligible_path = TABLE_DIR / "eo_to_bluesky_granger_input_daily_eligible.csv"
        detail_path = TABLE_DIR / "eo_to_bluesky_granger_fstats_detailed.csv"
    else:
        eligible_path = TABLE_DIR / "bluesky_to_eo_granger_input_daily_eligible.csv"
        detail_path = TABLE_DIR / "bluesky_to_eo_granger_fstats_detailed.csv"

    eligible_df.to_csv(eligible_path, index=False)
    detail_df.to_csv(detail_path, index=False)
    return eligible_df, detail_df


def plot_fstats(df: pd.DataFrame, stem: str, title: str) -> None:
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

    plot_df = df.copy()
    plot_df["relationship"] = pd.Categorical(plot_df["relationship"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("relationship")

    fig, ax = plt.subplots(figsize=(20, max(10, 0.72 * len(order))))
    sns.boxplot(
        data=plot_df,
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
    for _, row in plot_df.iterrows():
        y = y_positions[row["relationship"]] + rng.uniform(-0.12, 0.12)
        color = "#3182BD" if row["significant_005"] else "#E15759"
        ax.scatter(
            row["f_statistic"],
            y,
            s=40,
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

    ax.set_title(title)
    ax.set_xlabel("F-statistic")
    ax.set_ylabel("EO Theme | Bluesky Theme Mapping")
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3182BD", markeredgecolor="white", markersize=9, label="p-value < 0.05"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#E15759", markeredgecolor="white", markersize=9, label="p-value ≥ 0.05"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    fig.savefig(FIG_DIR / f"{stem}.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    write_crosswalk()
    bs_daily, eo_daily, crosswalk = load_sources()
    mapped_input = build_mapped_input(bs_daily, eo_daily, crosswalk)
    _, eo_to_bs = compute_granger(mapped_input, "eo_to_bluesky")
    _, bs_to_eo = compute_granger(mapped_input, "bluesky_to_eo")

    plot_fstats(
        eo_to_bs,
        "eo_to_bluesky_granger_fstats",
        "EO → Bluesky Granger Causality F-statistics (Daily Lags, 19 EO Themes)",
    )
    plot_fstats(
        bs_to_eo,
        "bluesky_to_eo_granger_fstats",
        "Bluesky → EO Granger Causality F-statistics (Daily Lags, 19 EO Themes)",
    )

    print(f"crosswalk\t{CROSSWALK_CSV}")
    print(f"input\t{TABLE_DIR / 'eo19_to_bluesky_granger_input_daily.csv'}")
    print(f"eo_input_eligible\t{TABLE_DIR / 'eo_to_bluesky_granger_input_daily_eligible.csv'}")
    print(f"bs_input_eligible\t{TABLE_DIR / 'bluesky_to_eo_granger_input_daily_eligible.csv'}")
    print(f"eo_table\t{TABLE_DIR / 'eo_to_bluesky_granger_fstats_detailed.csv'}")
    print(f"bs_table\t{TABLE_DIR / 'bluesky_to_eo_granger_fstats_detailed.csv'}")
    print(f"eo_figure\t{FIG_DIR / 'eo_to_bluesky_granger_fstats.png'}")
    print(f"bs_figure\t{FIG_DIR / 'bluesky_to_eo_granger_fstats.png'}")


if __name__ == "__main__":
    main()
