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
from statsmodels.tsa.stattools import adfuller, kpss

BASE_DIR = Path("local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs")
OUT_DIR = BASE_DIR / "stationarity_analysis"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "publication_figures"

INPUT_CSV = BASE_DIR / "tables" / "bluesky_theme_to_eo_granger_input_daily.csv"
ALPHA = 0.05

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")

def adf_test(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 20 or float(x.std(ddof=0)) == 0.0:
        return {
            "stat": np.nan,
            "pvalue": np.nan,
            "usedlag": np.nan,
            "nobs": len(x),
            "status": "not_testable",
        }
    try:
        stat, pvalue, usedlag, nobs, *_ = adfuller(x, autolag="AIC")
        return {
            "stat": float(stat),
            "pvalue": float(pvalue),
            "usedlag": int(usedlag),
            "nobs": int(nobs),
            "status": "stationary" if pvalue < ALPHA else "non_stationary",
        }
    except Exception:
        return {
            "stat": np.nan,
            "pvalue": np.nan,
            "usedlag": np.nan,
            "nobs": len(x),
            "status": "not_testable",
        }

def kpss_test(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 20 or float(x.std(ddof=0)) == 0.0:
        return {
            "stat": np.nan,
            "pvalue": np.nan,
            "usedlags": np.nan,
            "nobs": len(x),
            "status": "not_testable",
        }
    try:
        stat, pvalue, usedlags, *_ = kpss(x, regression="c", nlags="auto")
        return {
            "stat": float(stat),
            "pvalue": float(pvalue),
            "usedlags": int(usedlags),
            "nobs": len(x),
            "status": "stationary" if pvalue > ALPHA else "non_stationary",
        }
    except Exception:
        return {
            "stat": np.nan,
            "pvalue": np.nan,
            "usedlags": np.nan,
            "nobs": len(x),
            "status": "not_testable",
        }

def combined_decision(adf_status: str, kpss_status: str) -> str:
    if adf_status == "stationary" and kpss_status == "stationary":
        return "stationary"
    if adf_status == "non_stationary" and kpss_status == "non_stationary":
        return "non_stationary"
    if "not_testable" in {adf_status, kpss_status}:
        return "not_testable"
    return "mixed"

def test_series(series: pd.Series) -> dict:
    adf = adf_test(series)
    kpss_res = kpss_test(series)
    return {
        "adf_stat": adf["stat"],
        "adf_pvalue": adf["pvalue"],
        "adf_usedlag": adf["usedlag"],
        "adf_nobs": adf["nobs"],
        "adf_status": adf["status"],
        "kpss_stat": kpss_res["stat"],
        "kpss_pvalue": kpss_res["pvalue"],
        "kpss_usedlags": kpss_res["usedlags"],
        "kpss_nobs": kpss_res["nobs"],
        "kpss_status": kpss_res["status"],
        "combined_status": combined_decision(adf["status"], kpss_res["status"]),
    }

def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df["day"] = pd.to_datetime(df["day"])
    return df.sort_values(["bluesky_theme", "eo_theme", "day"]).reset_index(drop=True)

def run_tests(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    recommendations = []
    for (bluesky_theme, eo_theme), sub in df.groupby(["bluesky_theme", "eo_theme"], sort=True):
        sub = sub.sort_values("day").reset_index(drop=True)
        pair_id = f"{bluesky_theme} | {eo_theme}"
        series_map = {
            "eo_count": sub["eo_count"],
            "bluesky_posts": sub["bluesky_posts"],
        }
        pair_summary = {
            "bluesky_theme": bluesky_theme,
            "eo_theme": eo_theme,
            "pair_id": pair_id,
            "n_days": len(sub),
        }
        for var_name, s in series_map.items():
            level = test_series(s.astype(float))
            diff1 = test_series(s.diff().dropna().astype(float))
            pair_summary[f"{var_name}_level_status"] = level["combined_status"]
            pair_summary[f"{var_name}_diff1_status"] = diff1["combined_status"]
            pair_summary[f"{var_name}_level_adf_pvalue"] = level["adf_pvalue"]
            pair_summary[f"{var_name}_level_kpss_pvalue"] = level["kpss_pvalue"]
            pair_summary[f"{var_name}_diff1_adf_pvalue"] = diff1["adf_pvalue"]
            pair_summary[f"{var_name}_diff1_kpss_pvalue"] = diff1["kpss_pvalue"]
            if level["combined_status"] == "stationary":
                recommended = "level"
            elif diff1["combined_status"] == "stationary":
                recommended = "diff1"
            elif level["combined_status"] == "mixed" and diff1["combined_status"] == "stationary":
                recommended = "diff1"
            elif level["combined_status"] == "mixed":
                recommended = "inspect_mixed_level"
            elif diff1["combined_status"] == "mixed":
                recommended = "inspect_mixed_diff1"
            else:
                recommended = "not_testable"
            pair_summary[f"{var_name}_recommended_transform"] = recommended

            for transform_name, tested, nobs in [
                ("level", level, len(s.dropna())),
                ("diff1", diff1, len(s.diff().dropna())),
            ]:
                rows.append(
                    {
                        "bluesky_theme": bluesky_theme,
                        "eo_theme": eo_theme,
                        "pair_id": pair_id,
                        "variable": var_name,
                        "transform": transform_name,
                        "nobs_input": int(nobs),
                        **tested,
                    }
                )
        if (
            pair_summary["eo_count_recommended_transform"] == "level"
            and pair_summary["bluesky_posts_recommended_transform"] == "level"
        ):
            pair_summary["pair_granger_ready"] = "yes_level"
        elif (
            pair_summary["eo_count_recommended_transform"] in {"level", "diff1"}
            and pair_summary["bluesky_posts_recommended_transform"] in {"level", "diff1"}
        ):
            pair_summary["pair_granger_ready"] = "yes_with_transform"
        else:
            pair_summary["pair_granger_ready"] = "needs_manual_review"
        recommendations.append(pair_summary)
    return pd.DataFrame(rows), pd.DataFrame(recommendations)

def build_transformed_panel(df: pd.DataFrame, recommendations: pd.DataFrame) -> pd.DataFrame:
    merged = df.merge(
        recommendations[
            [
                "bluesky_theme",
                "eo_theme",
                "eo_count_recommended_transform",
                "bluesky_posts_recommended_transform",
                "pair_granger_ready",
            ]
        ],
        on=["bluesky_theme", "eo_theme"],
        how="left",
    )
    out_rows = []
    for (bluesky_theme, eo_theme), sub in merged.groupby(["bluesky_theme", "eo_theme"], sort=True):
        sub = sub.sort_values("day").copy()
        eo_transform = sub["eo_count_recommended_transform"].iloc[0]
        bs_transform = sub["bluesky_posts_recommended_transform"].iloc[0]
        if eo_transform == "diff1":
            sub["eo_count_stationary"] = sub["eo_count"].diff()
        else:
            sub["eo_count_stationary"] = sub["eo_count"]
        if bs_transform == "diff1":
            sub["bluesky_posts_stationary"] = sub["bluesky_posts"].diff()
        else:
            sub["bluesky_posts_stationary"] = sub["bluesky_posts"]
        sub["eo_count_transform_used"] = eo_transform
        sub["bluesky_posts_transform_used"] = bs_transform
        out_rows.append(sub)
    out = pd.concat(out_rows, ignore_index=True)
    return out

def plot_heatmap(pivot: pd.DataFrame, title: str, out_path: Path, cmap: str, center: float | None = None) -> None:
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(12, max(6, 0.55 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap=cmap,
        center=center,
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title, loc="left", fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("Theme Pair")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_status_heatmap(results: pd.DataFrame, out_path: Path) -> None:
    status_order = {"stationary": 2, "mixed": 1, "non_stationary": 0, "not_testable": -1}
    tmp = results.copy()
    tmp["status_score"] = tmp["combined_status"].map(status_order)
    tmp["series_key"] = tmp["variable"] + "_" + tmp["transform"]
    pivot = tmp.pivot(index="pair_id", columns="series_key", values="status_score").sort_index()
    sns.set_theme(style="white", context="talk")
    fig, ax = plt.subplots(figsize=(14, max(6, 0.55 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap=sns.color_palette(["#b2182b", "#fddbc7", "#d9f0d3", "#1a9850"], as_cmap=True),
        vmin=-1,
        vmax=2,
        linewidths=0.4,
        linecolor="white",
        cbar=False,
        ax=ax,
    )
    ax.set_title("Stationarity Decision by Pair and Series", loc="left", fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("Theme Pair")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def plot_recommendation_bar(recommendations: pd.DataFrame, out_path: Path) -> None:
    long = pd.concat(
        [
            recommendations[["pair_id", "eo_count_recommended_transform"]].rename(columns={"eo_count_recommended_transform": "recommended_transform"}).assign(variable="EO count"),
            recommendations[["pair_id", "bluesky_posts_recommended_transform"]].rename(columns={"bluesky_posts_recommended_transform": "recommended_transform"}).assign(variable="Bluesky posts"),
        ],
        ignore_index=True,
    )
    counts = (
        long.groupby(["variable", "recommended_transform"])
        .size()
        .reset_index(name="n")
    )
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=counts,
        x="recommended_transform",
        y="n",
        hue="variable",
        palette=["#c0392b", "#1f77b4"],
        ax=ax,
    )
    ax.set_title("Recommended Stationarity Transform Before Granger", loc="left", fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel("Number of series")
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    df = load_data()
    results, recommendations = run_tests(df)
    transformed = build_transformed_panel(df, recommendations)

    results = results.sort_values(["pair_id", "variable", "transform"]).reset_index(drop=True)
    recommendations = recommendations.sort_values(["pair_id"]).reset_index(drop=True)
    transformed = transformed.sort_values(["bluesky_theme", "eo_theme", "day"]).reset_index(drop=True)

    results.to_csv(TABLE_DIR / "stationarity_test_results.csv", index=False)
    recommendations.to_csv(TABLE_DIR / "stationarity_recommendations.csv", index=False)
    transformed.to_csv(TABLE_DIR / "granger_stationary_input_recommended.csv", index=False)

    adf_pivot = (
        results.assign(series_key=results["variable"] + "_" + results["transform"])
        .pivot(index="pair_id", columns="series_key", values="adf_pvalue")
        .sort_index()
    )
    kpss_pivot = (
        results.assign(series_key=results["variable"] + "_" + results["transform"])
        .pivot(index="pair_id", columns="series_key", values="kpss_pvalue")
        .sort_index()
    )
    plot_heatmap(adf_pivot, "ADF p-values by Pair and Series", FIG_DIR / "adf_pvalue_heatmap", cmap="Blues_r", center=None)
    plot_heatmap(kpss_pivot, "KPSS p-values by Pair and Series", FIG_DIR / "kpss_pvalue_heatmap", cmap="OrRd", center=None)
    plot_status_heatmap(results, FIG_DIR / "stationarity_status_heatmap")
    plot_recommendation_bar(recommendations, FIG_DIR / "recommended_transform_bar")

    print(f"results\t{TABLE_DIR / 'stationarity_test_results.csv'}")
    print(f"recommendations\t{TABLE_DIR / 'stationarity_recommendations.csv'}")
    print(f"stationary_input\t{TABLE_DIR / 'granger_stationary_input_recommended.csv'}")
    print(f"figures\t{FIG_DIR}")

if __name__ == "__main__":
    main()
