from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SRC_DIR = Path(
    "local_data/03_Github_data/10_Granger_causality_Bluesky_mapping_to_EOs/"
    "stationary_granger_analysis_corrected_May20/tables"
)
OUT_DIR = Path(
    "local_data/03_Github_data/11_Granger_causality_heatmaps"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)


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


def shorten_label(
    label: str,
    code_map: dict[str, str],
    short_map: dict[str, str],
) -> str:
    code = code_map.get(label, "")
    short = short_map.get(label, label)
    return f"{code}: {short}" if code else short


def pick_best_lag(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.sort_values(
        ["pair_id", "significant_005", "f_statistic", "p_value", "lag_days"],
        ascending=[True, False, False, True, True],
    )
    best = ranked.groupby("pair_id", as_index=False).first()
    best["cell_label"] = best.apply(
        lambda row: f"F={row['f_statistic']:.2f}\np={row['p_value']:.3f}\nD{int(row['lag_days'])}{'*' if row['significant_005'] else ''}",
        axis=1,
    )
    return best


def build_heatmap_inputs(
    df: pd.DataFrame,
    row_field: str,
    col_field: str,
    row_codes: dict[str, str],
    col_codes: dict[str, str],
    row_short: dict[str, str],
    col_short: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    plot_df = df.copy()
    plot_df["row_label"] = plot_df[row_field].map(
        lambda x: shorten_label(x, row_codes, row_short)
    )
    plot_df["col_label"] = plot_df[col_field].map(
        lambda x: shorten_label(x, col_codes, col_short)
    )

    value_matrix = plot_df.pivot(
        index="row_label",
        columns="col_label",
        values="f_statistic",
    )
    label_matrix = plot_df.pivot(
        index="row_label",
        columns="col_label",
        values="cell_label",
    )

    ordered_rows = [
        shorten_label(name, row_codes, row_short)
        for name in row_codes
        if shorten_label(name, row_codes, row_short) in value_matrix.index
    ]
    ordered_cols = [
        shorten_label(name, col_codes, col_short)
        for name in col_codes
        if shorten_label(name, col_codes, col_short) in value_matrix.columns
    ]

    value_matrix = value_matrix.reindex(index=ordered_rows, columns=ordered_cols)
    label_matrix = label_matrix.reindex(index=ordered_rows, columns=ordered_cols)
    return value_matrix, label_matrix


def plot_heatmap(
    value_matrix: pd.DataFrame,
    label_matrix: pd.DataFrame,
    title: str,
    out_stem: Path,
) -> None:
    sns.set_theme(style="white", context="talk")

    fig_width = max(14, 1.15 * len(value_matrix.columns))
    fig_height = max(8, 0.78 * len(value_matrix.index))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        value_matrix,
        annot=label_matrix,
        fmt="",
        cmap=cmap,
        linewidths=1.0,
        linecolor="white",
        cbar_kws={"label": "Best-lag Granger F-statistic"},
        annot_kws={"fontsize": 10, "fontweight": "bold"},
        ax=ax,
    )

    ax.set_title(title, loc="left", fontsize=20, pad=16)
    ax.set_xlabel("Bluesky themes" if "Bluesky" in value_matrix.columns.name else value_matrix.columns.name or "")
    ax.set_ylabel("EO themes" if "EO" in value_matrix.index.name else value_matrix.index.name or "")
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)

    note = "Cell text shows strongest F-statistic, p-value, and best lag. `*` marks p < 0.05."
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=10)

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_stem.with_suffix(".png"), dpi=500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    bluesky_to_eo = pd.read_csv(SRC_DIR / "bluesky_to_eo_granger_fstats_stationary.csv")
    eo_to_bluesky = pd.read_csv(SRC_DIR / "eo_to_bluesky_granger_fstats_stationary.csv")

    b2e_best = pick_best_lag(bluesky_to_eo)
    e2b_best = pick_best_lag(eo_to_bluesky)

    b2e_best.to_csv(OUT_DIR / "bluesky_to_eo_best_lag_summary.csv", index=False)
    e2b_best.to_csv(OUT_DIR / "eo_to_bluesky_best_lag_summary.csv", index=False)

    eo_rows, bluesky_cols = build_heatmap_inputs(
        e2b_best,
        row_field="eo_theme",
        col_field="bluesky_theme",
        row_codes=EO_CODE_BY_NAME,
        col_codes=THEME_CODE_BY_NAME,
        row_short=EO_SHORT_BY_NAME,
        col_short=THEME_SHORT_BY_NAME,
    )
    eo_rows.index.name = "EO themes"
    eo_rows.columns.name = "Bluesky themes"

    eo_labels = build_heatmap_inputs(
        e2b_best,
        row_field="eo_theme",
        col_field="bluesky_theme",
        row_codes=EO_CODE_BY_NAME,
        col_codes=THEME_CODE_BY_NAME,
        row_short=EO_SHORT_BY_NAME,
        col_short=THEME_SHORT_BY_NAME,
    )[1]
    eo_labels.index.name = "EO themes"
    eo_labels.columns.name = "Bluesky themes"

    bluesky_rows, eo_cols = build_heatmap_inputs(
        b2e_best,
        row_field="bluesky_theme",
        col_field="eo_theme",
        row_codes=THEME_CODE_BY_NAME,
        col_codes=EO_CODE_BY_NAME,
        row_short=THEME_SHORT_BY_NAME,
        col_short=EO_SHORT_BY_NAME,
    )
    bluesky_rows.index.name = "Bluesky themes"
    bluesky_rows.columns.name = "EO themes"

    bluesky_labels = build_heatmap_inputs(
        b2e_best,
        row_field="bluesky_theme",
        col_field="eo_theme",
        row_codes=THEME_CODE_BY_NAME,
        col_codes=EO_CODE_BY_NAME,
        row_short=THEME_SHORT_BY_NAME,
        col_short=EO_SHORT_BY_NAME,
    )[1]
    bluesky_labels.index.name = "Bluesky themes"
    bluesky_labels.columns.name = "EO themes"

    plot_heatmap(
        eo_rows,
        eo_labels,
        "EO to Bluesky Granger causality heatmap",
        OUT_DIR / "eo_to_bluesky_granger_heatmap",
    )
    plot_heatmap(
        bluesky_rows,
        bluesky_labels,
        "Bluesky to EO Granger causality heatmap",
        OUT_DIR / "bluesky_to_eo_granger_heatmap",
    )


if __name__ == "__main__":
    main()
