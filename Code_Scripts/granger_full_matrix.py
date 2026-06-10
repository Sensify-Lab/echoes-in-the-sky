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
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, kpss

BS_DAILY_CSV = Path(
    "local_data/03_Github_data/06_EO_effect_bluesky/tables/bluesky_theme_daily.csv"
)
EO_DAILY_CSV = Path(
    "local_data/03_Github_data/06_EO_effect_bluesky/tables/eo_theme_daily.csv"
)
OUT_DIR = Path(
    "local_data/03_Github_data/11_Granger_causality_heatmaps/full_matrix_strict_stationary_only_no_na_sigcount"
)
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR = OUT_DIR / "figures"

WINDOW_START = "2025-01-20"
WINDOW_END = "2026-02-01"
ALPHA = 0.05
MAX_LAG = 14

THEME_CODE_BY_NAME = {
    "Executive Power, Patronage, and Administrative Control": "T1",
    "Republican, MAGA, and Partisan Conflict": "T2",
    "National Security, Military, and Geopolitical Conflict": "T3",
    "Authoritarianism, Fascism, and Democratic Erosion": "T4",
    "Media Narratives, Messaging, and Public Framing": "T5",
    "Legal Accountability, Courts, and Constitutional Conflict": "T6",
    "Economic Policy, Taxation, and Distributional Conflict": "T7",
    "Technology, Platforms, and Digital Power": "T8",
    "Public Lands, Environment, and Climate Governance": "T9",
    "Education, Universities, and Intellectual Control": "T10",
    "Health, Public Health, and Bodily Politics": "T11",
    "Immigration, Borders, and Sovereignty": "T12",
    "Aesthetics, Symbolism, and Performance of Power": "T13",
    "Race, Civil Rights, and Identity Hierarchies": "T14",
}

EO_CODE_BY_NAME = {
    "Executive Coordination and Administrative Task Forces": "E1",
    "Public Order, Crime, and Border Enforcement": "E2",
    "Foreign Policy and External Relations": "E3",
    "National Security, Defense, and Strategic Resilience": "E4",
    "Trade, Tariffs, and Export Promotion": "E5",
    "Civil Rights, Equality, and Gender Policy": "E6",
    "Domestic Investment and Market Competition": "E7",
    "Health, Reproductive Care, and Medical Research": "E8",
    "Education and Higher-Ed Transparency": "E9",
    "Energy Development and Resource Expansion": "E10",
    "Technology, Science, and Artificial Intelligence Development": "E11",
    "Water Supply and Disaster Response": "E12",
    "Cultural Heritage and National Pastime": "E13",
    "Environmental Stewardship and Beautification": "E14",
    "Housing Affordability and Development Deregulation": "E15",
    "Infrastructure and Maritime Transportation Rules": "E16",
    "Public Lands, Parks, and Timber Production": "E17",
    "Family Support and Child Wellbeing": "E18",
    "Workforce, Training, and Retirement Security": "E19",
}

THEME_SHORT_BY_NAME = {
    "Executive Power, Patronage, and Administrative Control": "Executive control",
    "Republican, MAGA, and Partisan Conflict": "Partisan conflict",
    "National Security, Military, and Geopolitical Conflict": "National security",
    "Authoritarianism, Fascism, and Democratic Erosion": "Democratic erosion",
    "Media Narratives, Messaging, and Public Framing": "Media framing",
    "Legal Accountability, Courts, and Constitutional Conflict": "Legal conflict",
    "Economic Policy, Taxation, and Distributional Conflict": "Economic policy",
    "Technology, Platforms, and Digital Power": "Digital power",
    "Public Lands, Environment, and Climate Governance": "Climate governance",
    "Education, Universities, and Intellectual Control": "Education control",
    "Health, Public Health, and Bodily Politics": "Public health",
    "Immigration, Borders, and Sovereignty": "Immigration",
    "Aesthetics, Symbolism, and Performance of Power": "Symbolic power",
    "Race, Civil Rights, and Identity Hierarchies": "Civil rights and race",
}

EO_SHORT_BY_NAME = {
    "Executive Coordination and Administrative Task Forces": "Executive coordination",
    "Public Order, Crime, and Border Enforcement": "Border enforcement",
    "Foreign Policy and External Relations": "Foreign policy",
    "National Security, Defense, and Strategic Resilience": "Defense and resilience",
    "Trade, Tariffs, and Export Promotion": "Trade and tariffs",
    "Civil Rights, Equality, and Gender Policy": "Civil rights policy",
    "Domestic Investment and Market Competition": "Domestic investment",
    "Health, Reproductive Care, and Medical Research": "Health and care",
    "Education and Higher-Ed Transparency": "Higher-ed transparency",
    "Energy Development and Resource Expansion": "Energy expansion",
    "Technology, Science, and Artificial Intelligence Development": "Tech and AI",
    "Water Supply and Disaster Response": "Water and disaster",
    "Cultural Heritage and National Pastime": "Cultural heritage",
    "Environmental Stewardship and Beautification": "Environmental stewardship",
    "Housing Affordability and Development Deregulation": "Housing deregulation",
    "Infrastructure and Maritime Transportation Rules": "Infrastructure rules",
    "Public Lands, Parks, and Timber Production": "Public lands",
    "Family Support and Child Wellbeing": "Family support",
    "Workforce, Training, and Retirement Security": "Workforce security",
}

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def adf_test(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 20 or float(x.std(ddof=0)) == 0.0:
        return {"status": "not_testable", "pvalue": np.nan}
    try:
        pvalue = float(adfuller(x, autolag="AIC")[1])
        return {"status": "stationary" if pvalue < ALPHA else "non_stationary", "pvalue": pvalue}
    except Exception:
        return {"status": "not_testable", "pvalue": np.nan}

def kpss_test(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 20 or float(x.std(ddof=0)) == 0.0:
        return {"status": "not_testable", "pvalue": np.nan}
    try:
        pvalue = float(kpss(x, regression="c", nlags="auto")[1])
        return {"status": "stationary" if pvalue > ALPHA else "non_stationary", "pvalue": pvalue}
    except Exception:
        return {"status": "not_testable", "pvalue": np.nan}

def combined_decision(adf_status: str, kpss_status: str) -> str:
    if adf_status == "stationary" and kpss_status == "stationary":
        return "stationary"
    if adf_status == "non_stationary" and kpss_status == "non_stationary":
        return "non_stationary"
    if "not_testable" in {adf_status, kpss_status}:
        return "not_testable"
    return "mixed"

def series_stationarity(series: pd.Series) -> dict:
    level_adf = adf_test(series.astype(float))
    level_kpss = kpss_test(series.astype(float))
    diff = series.diff().dropna().astype(float)
    diff_adf = adf_test(diff)
    diff_kpss = kpss_test(diff)
    level_status = combined_decision(level_adf["status"], level_kpss["status"])
    diff1_status = combined_decision(diff_adf["status"], diff_kpss["status"])

    if level_status == "stationary":
        recommended = "level"
    elif diff1_status == "stationary":
        recommended = "diff1"
    elif level_status == "mixed" and diff1_status == "stationary":
        recommended = "diff1"
    elif level_status == "mixed":
        recommended = "inspect_mixed_level"
    elif diff1_status == "mixed":
        recommended = "inspect_mixed_diff1"
    else:
        recommended = "not_testable"

    return {
        "level_status": level_status,
        "diff1_status": diff1_status,
        "level_adf_pvalue": level_adf["pvalue"],
        "level_kpss_pvalue": level_kpss["pvalue"],
        "diff1_adf_pvalue": diff_adf["pvalue"],
        "diff1_kpss_pvalue": diff_kpss["pvalue"],
        "recommended_transform": recommended,
    }

def load_series() -> tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    bs = pd.read_csv(BS_DAILY_CSV)
    eo = pd.read_csv(EO_DAILY_CSV)
    bs["day"] = pd.to_datetime(bs["day"])
    eo["day"] = pd.to_datetime(eo["day"])
    idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    return bs, eo, idx

def build_full_panel(
    bs_daily: pd.DataFrame,
    eo_daily: pd.DataFrame,
    idx: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bs_themes = list(THEME_CODE_BY_NAME.keys())
    eo_themes = list(EO_CODE_BY_NAME.keys())

    bs_series = {}
    eo_series = {}
    bs_meta_rows = []
    eo_meta_rows = []

    for bs_theme in bs_themes:
        s = (
            bs_daily[bs_daily["bluesky_theme"] == bs_theme]
            .groupby("day")["bluesky_posts"]
            .sum()
            .reindex(idx, fill_value=0)
            .astype(float)
        )
        meta = series_stationarity(s)
        bs_series[bs_theme] = s
        bs_meta_rows.append({"bluesky_theme": bs_theme, **meta})

    for eo_theme in eo_themes:
        s = (
            eo_daily[eo_daily["eo_theme"] == eo_theme]
            .groupby("day")["eo_count"]
            .sum()
            .reindex(idx, fill_value=0)
            .astype(float)
        )
        meta = series_stationarity(s)
        eo_series[eo_theme] = s
        eo_meta_rows.append({"eo_theme": eo_theme, **meta})

    bs_meta = pd.DataFrame(bs_meta_rows)
    eo_meta = pd.DataFrame(eo_meta_rows)
    bs_lookup = bs_meta.set_index("bluesky_theme").to_dict(orient="index")
    eo_lookup = eo_meta.set_index("eo_theme").to_dict(orient="index")

    rows = []
    for bs_theme in bs_themes:
        for eo_theme in eo_themes:
            bs_transform = bs_lookup[bs_theme]["recommended_transform"]
            eo_transform = eo_lookup[eo_theme]["recommended_transform"]
            pair_ready = bs_transform in {"level", "diff1"} and eo_transform in {"level", "diff1"}
            panel = pd.DataFrame(
                {
                    "day": idx,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "eo_count": eo_series[eo_theme].values,
                    "bluesky_posts": bs_series[bs_theme].values,
                    "eo_count_transform_used": eo_transform,
                    "bluesky_posts_transform_used": bs_transform,
                    "pair_ready_strict": pair_ready,
                }
            )
            if eo_transform == "diff1":
                panel["eo_count_stationary"] = panel["eo_count"].diff()
            elif eo_transform == "level":
                panel["eo_count_stationary"] = panel["eo_count"]
            else:
                panel["eo_count_stationary"] = np.nan

            if bs_transform == "diff1":
                panel["bluesky_posts_stationary"] = panel["bluesky_posts"].diff()
            elif bs_transform == "level":
                panel["bluesky_posts_stationary"] = panel["bluesky_posts"]
            else:
                panel["bluesky_posts_stationary"] = np.nan
            rows.append(panel)

    return pd.concat(rows, ignore_index=True), bs_meta, eo_meta

def run_granger_strict(full_panel: pd.DataFrame, direction: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    skipped = []
    for (bs_theme, eo_theme), sub in full_panel.groupby(["bluesky_theme", "eo_theme"], sort=True):
        sub = sub.sort_values("day").copy()
        pair_id = f"{bs_theme} | {eo_theme}"

        if not bool(sub["pair_ready_strict"].iloc[0]):
            skipped.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "direction": direction,
                    "reason": "not_strictly_stationary_correctable",
                    "eo_transform_used": sub["eo_count_transform_used"].iloc[0],
                    "bluesky_transform_used": sub["bluesky_posts_transform_used"].iloc[0],
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
            skipped.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "direction": direction,
                    "reason": "insufficient_rows_after_strict_transform",
                    "eo_transform_used": sub["eo_count_transform_used"].iloc[0],
                    "bluesky_transform_used": sub["bluesky_posts_transform_used"].iloc[0],
                }
            )
            continue

        if any(float(sub2[col].std(ddof=0)) == 0.0 for col in sub2.columns):
            skipped.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "direction": direction,
                    "reason": "constant_series_after_strict_transform",
                    "eo_transform_used": sub["eo_count_transform_used"].iloc[0],
                    "bluesky_transform_used": sub["bluesky_posts_transform_used"].iloc[0],
                }
            )
            continue

        try:
            tests = grangercausalitytests(sub2, maxlag=MAX_LAG, verbose=False)
        except Exception as exc:
            skipped.append(
                {
                    "pair_id": pair_id,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "direction": direction,
                    "reason": f"granger_error:{type(exc).__name__}",
                    "eo_transform_used": sub["eo_count_transform_used"].iloc[0],
                    "bluesky_transform_used": sub["bluesky_posts_transform_used"].iloc[0],
                }
            )
            continue

        for lag, payload in tests.items():
            ftest = payload[0]["ssr_ftest"]
            rows.append(
                {
                    "pair_id": pair_id,
                    "direction": direction,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "tested_var": tested_var,
                    "target_var": target_var,
                    "lag_days": int(lag),
                    "f_statistic": float(ftest[0]),
                    "p_value": float(ftest[1]),
                    "df_denom": float(ftest[2]),
                    "df_num": float(ftest[3]),
                    "significant_005": bool(ftest[1] < 0.05),
                    "eo_transform_used": sub["eo_count_transform_used"].iloc[0],
                    "bluesky_transform_used": sub["bluesky_posts_transform_used"].iloc[0],
                    "nobs_after_transform": int(len(sub2)),
                }
            )

    return pd.DataFrame(rows), pd.DataFrame(skipped)

def significance_stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""

def build_complete_best_lag(
    fstats_df: pd.DataFrame,
    skipped_df: pd.DataFrame,
    direction: str,
) -> pd.DataFrame:
    if not fstats_df.empty:
        ranked = fstats_df.sort_values(
            ["pair_id", "significant_005", "f_statistic", "p_value", "lag_days"],
            ascending=[True, False, False, True, True],
        )
        best = ranked.groupby("pair_id", as_index=False).first()
    else:
        best = pd.DataFrame(columns=fstats_df.columns)

    if not fstats_df.empty:
        sig_counts = (
            fstats_df.assign(is_sig=fstats_df["p_value"] < 0.05)
            .groupby("pair_id", as_index=False)["is_sig"]
            .sum()
            .rename(columns={"is_sig": "sig_lag_count"})
        )
        best = best.merge(sig_counts, on="pair_id", how="left")

        earliest_sig = (
            fstats_df.loc[fstats_df["p_value"] < 0.05]
            .sort_values(["pair_id", "lag_days"])
            .groupby("pair_id", as_index=False)
            .first()[["pair_id", "lag_days", "f_statistic", "p_value"]]
            .rename(
                columns={
                    "lag_days": "earliest_sig_lag_days",
                    "f_statistic": "earliest_sig_f_statistic",
                    "p_value": "earliest_sig_lag_p_value",
                }
            )
        )
        best = best.merge(earliest_sig, on="pair_id", how="left")
    else:
        best["sig_lag_count"] = []
        best["earliest_sig_lag_days"] = []
        best["earliest_sig_f_statistic"] = []
        best["earliest_sig_lag_p_value"] = []

    best["sig_lag_count"] = best["sig_lag_count"].fillna(0).astype(int)
    best["earliest_sig_lag_label"] = best.apply(
        lambda row: (
            f"D{int(row['earliest_sig_lag_days'])}"
            f"{significance_stars(row['earliest_sig_lag_p_value'])}"
        )
        if pd.notna(row["earliest_sig_lag_days"])
        else "",
        axis=1,
    )
    best["cell_label"] = (
        best.apply(
            lambda row: (
                f"F={row['f_statistic']:.1f}\n"
                f"D{int(row['lag_days'])}{significance_stars(row['p_value'])}\n"
                f"n={int(row['sig_lag_count']):02d}"
            )
            if row["p_value"] < 0.05
            else f"F={row['f_statistic']:.1f}",
            axis=1,
        )
        if not best.empty
        else []
    )
    best["earliest_sig_cell_label"] = (
        best.apply(
            lambda row: (
                f"F={row['earliest_sig_f_statistic']:.1f}\n"
                f"{row['earliest_sig_lag_label']}"
            )
            if row["p_value"] < 0.05
            else f"F={row['f_statistic']:.1f}",
            axis=1,
        )
        if not best.empty
        else []
    )
    best["strict_na"] = False

    all_rows = []
    present = set(best["pair_id"].tolist()) if not best.empty else set()
    skipped_lookup = {}
    if not skipped_df.empty:
        skipped_lookup = skipped_df.set_index("pair_id").to_dict(orient="index")

    for bs_theme in THEME_CODE_BY_NAME:
        for eo_theme in EO_CODE_BY_NAME:
            pair_id = f"{bs_theme} | {eo_theme}"
            if pair_id in present:
                continue
            info = skipped_lookup.get(pair_id, {})
            all_rows.append(
                {
                    "pair_id": pair_id,
                    "direction": direction,
                    "bluesky_theme": bs_theme,
                    "eo_theme": eo_theme,
                    "tested_var": "NA",
                    "target_var": "NA",
                    "lag_days": np.nan,
                    "f_statistic": np.nan,
                    "p_value": np.nan,
                    "df_denom": np.nan,
                    "df_num": np.nan,
                    "significant_005": False,
                    "eo_transform_used": info.get("eo_transform_used"),
                    "bluesky_transform_used": info.get("bluesky_transform_used"),
                    "nobs_after_transform": np.nan,
                    "sig_lag_count": 0,
                    "earliest_sig_lag_days": np.nan,
                    "earliest_sig_f_statistic": np.nan,
                    "earliest_sig_lag_p_value": np.nan,
                    "earliest_sig_lag_label": "",
                    "cell_label": "",
                    "earliest_sig_cell_label": "",
                    "strict_na": True,
                }
            )

    if all_rows:
        best = pd.concat([best, pd.DataFrame(all_rows)], ignore_index=True)
    return best

def shorten_label(label: str, code_map: dict[str, str], short_map: dict[str, str]) -> str:
    return f"{code_map[label]}: {short_map[label]}"

def build_heatmap_inputs(
    df: pd.DataFrame,
    row_field: str,
    col_field: str,
    row_codes: dict[str, str],
    col_codes: dict[str, str],
    row_short: dict[str, str],
    col_short: dict[str, str],
    label_field: str = "cell_label",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plot_df = df.copy()
    plot_df["row_label"] = plot_df[row_field].map(lambda x: shorten_label(x, row_codes, row_short))
    plot_df["col_label"] = plot_df[col_field].map(lambda x: shorten_label(x, col_codes, col_short))
    values = plot_df.pivot(index="row_label", columns="col_label", values="f_statistic")
    labels = plot_df.pivot(index="row_label", columns="col_label", values=label_field)
    pvalues = plot_df.pivot(index="row_label", columns="col_label", values="p_value")
    row_order = [shorten_label(x, row_codes, row_short) for x in row_codes]
    col_order = [shorten_label(x, col_codes, col_short) for x in col_codes]
    values = values.reindex(index=row_order, columns=col_order)
    labels = labels.reindex(index=row_order, columns=col_order)
    pvalues = pvalues.reindex(index=row_order, columns=col_order)
    values = values.dropna(axis=0, how="all").dropna(axis=1, how="all")
    labels = labels.reindex(index=values.index, columns=values.columns).fillna("")
    pvalues = pvalues.reindex(index=values.index, columns=values.columns)
    return values, labels, pvalues

def build_earliest_sig_heatmap_inputs(
    df: pd.DataFrame,
    row_field: str,
    col_field: str,
    row_codes: dict[str, str],
    col_codes: dict[str, str],
    row_short: dict[str, str],
    col_short: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plot_df = df.copy()
    plot_df["row_label"] = plot_df[row_field].map(lambda x: shorten_label(x, row_codes, row_short))
    plot_df["col_label"] = plot_df[col_field].map(lambda x: shorten_label(x, col_codes, col_short))
    values = plot_df.pivot(index="row_label", columns="col_label", values="earliest_sig_f_statistic")
    labels = plot_df.pivot(index="row_label", columns="col_label", values="earliest_sig_cell_label")
    pvalues = plot_df.pivot(index="row_label", columns="col_label", values="earliest_sig_lag_p_value")
    row_order = [shorten_label(x, row_codes, row_short) for x in row_codes]
    col_order = [shorten_label(x, col_codes, col_short) for x in col_codes]
    values = values.reindex(index=row_order, columns=col_order)
    labels = labels.reindex(index=row_order, columns=col_order)
    pvalues = pvalues.reindex(index=row_order, columns=col_order)
    values = values.dropna(axis=0, how="all").dropna(axis=1, how="all")
    labels = labels.reindex(index=values.index, columns=values.columns).fillna("")
    pvalues = pvalues.reindex(index=values.index, columns=values.columns)
    return values, labels, pvalues

def plot_heatmap(
    values: pd.DataFrame,
    labels: pd.DataFrame,
    pvalues: pd.DataFrame,
    title: str,
    out_stem: Path,
) -> None:
    sns.set_theme(style="white", context="talk")
    fig_width = max(18, 0.9 * len(values.columns))
    fig_height = max(12, 0.62 * len(values.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    na_mask = values.isna()
    vmin = np.nanmin(values.to_numpy())
    vmax = np.nanmax(values.to_numpy())

    sns.heatmap(
        values,
        mask=na_mask | (pvalues <= 0.05),
        cmap="Greys",
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        linewidths=0.7,
        linecolor="white",
        ax=ax,
    )
    sns.heatmap(
        values,
        mask=na_mask | (pvalues > 0.05),
        cmap="Reds",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "F-statistic", "shrink": 0.82},
        linewidths=0.7,
        linecolor="white",
        ax=ax,
    )
    sns.heatmap(
        values,
        mask=na_mask,
        annot=labels,
        fmt="",
        cmap=sns.color_palette([(1, 1, 1, 0)], as_cmap=True),
        cbar=False,
        linewidths=0.7,
        linecolor="white",
        annot_kws={"fontsize": 6.2, "fontweight": "bold"},
        ax=ax,
    )
    ax.set_title(title, loc="left", fontsize=18, pad=14)
    ax.set_xlabel(values.columns.name or "")
    ax.set_ylabel(values.index.name or "")
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.text(
        0.01,
        0.01,
        "Strict stationary-only rerun. Grey cells have non-significant best lags; red cells have significant best lags. Significant cells show D# and n = number of significant lags from D1 to D14.",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    bs_daily, eo_daily, idx = load_series()
    full_panel, bs_meta, eo_meta = build_full_panel(bs_daily, eo_daily, idx)

    b2e_fstats, b2e_skipped = run_granger_strict(full_panel, "bluesky_to_eo")
    e2b_fstats, e2b_skipped = run_granger_strict(full_panel, "eo_to_bluesky")
    b2e_best = build_complete_best_lag(b2e_fstats, b2e_skipped, "bluesky_to_eo")
    e2b_best = build_complete_best_lag(e2b_fstats, e2b_skipped, "eo_to_bluesky")

    full_panel.to_csv(TABLE_DIR / "full_matrix_granger_input_daily_strict.csv", index=False)
    bs_meta.to_csv(TABLE_DIR / "bluesky_theme_stationarity_metadata_strict.csv", index=False)
    eo_meta.to_csv(TABLE_DIR / "eo_theme_stationarity_metadata_strict.csv", index=False)
    b2e_fstats.to_csv(TABLE_DIR / "bluesky_to_eo_granger_fstats_strict.csv", index=False)
    e2b_fstats.to_csv(TABLE_DIR / "eo_to_bluesky_granger_fstats_strict.csv", index=False)
    b2e_skipped.to_csv(TABLE_DIR / "bluesky_to_eo_skipped_pairs_strict.csv", index=False)
    e2b_skipped.to_csv(TABLE_DIR / "eo_to_bluesky_skipped_pairs_strict.csv", index=False)
    b2e_best.to_csv(TABLE_DIR / "bluesky_to_eo_best_lag_strict.csv", index=False)
    e2b_best.to_csv(TABLE_DIR / "eo_to_bluesky_best_lag_strict.csv", index=False)

    e2b_values, e2b_labels, e2b_pvalues = build_heatmap_inputs(
        e2b_best,
        row_field="eo_theme",
        col_field="bluesky_theme",
        row_codes=EO_CODE_BY_NAME,
        col_codes=THEME_CODE_BY_NAME,
        row_short=EO_SHORT_BY_NAME,
        col_short=THEME_SHORT_BY_NAME,
    )
    e2b_earliest_values, e2b_earliest_labels, e2b_earliest_pvalues = build_earliest_sig_heatmap_inputs(
        e2b_best,
        row_field="eo_theme",
        col_field="bluesky_theme",
        row_codes=EO_CODE_BY_NAME,
        col_codes=THEME_CODE_BY_NAME,
        row_short=EO_SHORT_BY_NAME,
        col_short=THEME_SHORT_BY_NAME,
    )
    e2b_values.index.name = "EO themes"
    e2b_values.columns.name = "Bluesky themes"

    b2e_values, b2e_labels, b2e_pvalues = build_heatmap_inputs(
        b2e_best,
        row_field="bluesky_theme",
        col_field="eo_theme",
        row_codes=THEME_CODE_BY_NAME,
        col_codes=EO_CODE_BY_NAME,
        row_short=THEME_SHORT_BY_NAME,
        col_short=EO_SHORT_BY_NAME,
    )
    b2e_earliest_values, b2e_earliest_labels, b2e_earliest_pvalues = build_earliest_sig_heatmap_inputs(
        b2e_best,
        row_field="bluesky_theme",
        col_field="eo_theme",
        row_codes=THEME_CODE_BY_NAME,
        col_codes=EO_CODE_BY_NAME,
        row_short=THEME_SHORT_BY_NAME,
        col_short=EO_SHORT_BY_NAME,
    )
    b2e_values.index.name = "Bluesky themes"
    b2e_values.columns.name = "EO themes"

    plot_heatmap(
        e2b_values,
        e2b_labels,
        e2b_pvalues,
        "EO to Bluesky Granger causality heatmap (strict stationary only, significance color grid)",
        FIG_DIR / "eo_to_bluesky_granger_heatmap_strict_sigcolor_nlags",
    )
    plot_heatmap(
        b2e_values,
        b2e_labels,
        b2e_pvalues,
        "Bluesky to EO Granger causality heatmap (strict stationary only, significance color grid)",
        FIG_DIR / "bluesky_to_eo_granger_heatmap_strict_sigcolor_nlags",
    )
    plot_heatmap(
        e2b_earliest_values,
        e2b_earliest_labels,
        e2b_earliest_pvalues,
        "EO to Bluesky Granger causality heatmap (strict stationary only, earliest significant lag)",
        FIG_DIR / "eo_to_bluesky_granger_heatmap_strict_sigcolor_earliestlag",
    )
    plot_heatmap(
        b2e_earliest_values,
        b2e_earliest_labels,
        b2e_earliest_pvalues,
        "Bluesky to EO Granger causality heatmap (strict stationary only, earliest significant lag)",
        FIG_DIR / "bluesky_to_eo_granger_heatmap_strict_sigcolor_earliestlag",
    )

if __name__ == "__main__":
    main()
