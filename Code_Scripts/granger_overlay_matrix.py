from __future__ import annotations

from pathlib import Path
import json

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

SOURCE_DIR = Path(
    "local_data/03_Github_data/11_Granger_causality_heatmaps/"
    "full_matrix_strict_stationary_only_no_na_sigcount"
)
INPUT_CSV = SOURCE_DIR / "tables" / "full_matrix_granger_input_daily_strict.csv"

BASE_DIR = SOURCE_DIR / "corrected_overlay_matrix_from_final_figure7"
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"

WINDOW_START = pd.Timestamp("2025-01-20")
WINDOW_END = pd.Timestamp("2026-02-01")

BLUESKY_COLOR = "#1f5fa8"
EO_COLOR = "#8e2a1e"
ZERO_COLOR = "#b0b0b0"
GRID_COLOR = "#e5e5e5"

CELL_WIDTH = 1.55
CELL_HEIGHT = 1.18
FIG_DPI = 320

THEME_CODE_BY_NAME = {
    "Executive Power, Patronage, and Administrative Control": "B1",
    "Republican, MAGA, and Partisan Conflict": "B2",
    "National Security, Military, and Geopolitical Conflict": "B3",
    "Authoritarianism, Fascism, and Democratic Erosion": "B4",
    "Media Narratives, Messaging, and Public Framing": "B5",
    "Legal Accountability, Courts, and Constitutional Conflict": "B6",
    "Economic Policy, Taxation, and Distributional Conflict": "B7",
    "Technology, Platforms, and Digital Power": "B8",
    "Public Lands, Environment, and Climate Governance": "B9",
    "Education, Universities, and Intellectual Control": "B10",
    "Health, Public Health, and Bodily Politics": "B11",
    "Immigration, Borders, and Sovereignty": "B12",
    "Aesthetics, Symbolism, and Performance of Power": "B13",
    "Race, Civil Rights, and Identity Hierarchies": "B14",
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

BLUESKY_THEME_ORDER = list(THEME_CODE_BY_NAME.keys())
EO_THEME_ORDER = list(EO_CODE_BY_NAME.keys())

def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

def load_panel() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV, parse_dates=["day"])
    df = df.sort_values(["eo_theme", "bluesky_theme", "day"]).reset_index(drop=True)
    df.to_csv(TABLE_DIR / "full_matrix_granger_input_daily_strict_copy.csv", index=False)
    return df

def write_abbreviation_tables() -> None:
    pd.DataFrame(
        [{"code": code, "bluesky_theme": name} for name, code in THEME_CODE_BY_NAME.items()]
    ).to_csv(TABLE_DIR / "bluesky_theme_abbreviations.csv", index=False)
    pd.DataFrame(
        [{"code": code, "eo_theme": name} for name, code in EO_CODE_BY_NAME.items()]
    ).to_csv(TABLE_DIR / "eo_theme_abbreviations.csv", index=False)

def build_matrix_figure(panel: pd.DataFrame) -> None:
    bluesky_present = set(panel["bluesky_theme"].dropna().unique())
    eo_present = set(panel["eo_theme"].dropna().unique())
    bluesky_themes = [theme for theme in BLUESKY_THEME_ORDER if theme in bluesky_present]
    eo_themes = [theme for theme in EO_THEME_ORDER if theme in eo_present]

    fig_w = len(bluesky_themes) * CELL_WIDTH
    fig_h = len(eo_themes) * CELL_HEIGHT
    fig, axes = plt.subplots(
        len(eo_themes),
        len(bluesky_themes),
        figsize=(fig_w, fig_h),
        dpi=FIG_DPI,
        sharex=True,
        squeeze=False,
    )

    for r, eo_theme in enumerate(eo_themes):
        for c, bluesky_theme in enumerate(bluesky_themes):
            ax = axes[r, c]
            sub = panel[
                (panel["bluesky_theme"] == bluesky_theme)
                & (panel["eo_theme"] == eo_theme)
            ].copy()
            ax2 = ax.twinx()
            bs_max_abs = float(sub["bluesky_posts_stationary"].abs().max(skipna=True))
            eo_max_abs = float(sub["eo_count_stationary"].abs().max(skipna=True))
            if not pd.notna(bs_max_abs) or bs_max_abs == 0:
                bs_max_abs = 1.0
            if not pd.notna(eo_max_abs) or eo_max_abs == 0:
                eo_max_abs = 1.0

            ax.plot(
                sub["day"],
                sub["bluesky_posts_stationary"],
                color=BLUESKY_COLOR,
                linewidth=0.55,
                alpha=0.95,
            )
            ax2.plot(
                sub["day"],
                sub["eo_count_stationary"],
                color=EO_COLOR,
                linewidth=0.55,
                alpha=0.95,
            )
            ax.axhline(0, color=ZERO_COLOR, linewidth=0.45, linestyle="--")
            ax2.axhline(0, color=ZERO_COLOR, linewidth=0.45, linestyle="--")
            ax.grid(axis="y", color=GRID_COLOR, linewidth=0.35, alpha=0.7)
            ax.set_xlim(WINDOW_START, WINDOW_END)
            ax2.set_xlim(WINDOW_START, WINDOW_END)
            ax.set_ylim(-bs_max_abs, bs_max_abs)
            ax2.set_ylim(-eo_max_abs, eo_max_abs)

            if r == 0:
                ax.set_title(THEME_CODE_BY_NAME.get(bluesky_theme, bluesky_theme), fontsize=8.2, fontweight="bold", pad=4)
            if c == 0:
                ax.set_ylabel(EO_CODE_BY_NAME.get(eo_theme, eo_theme), fontsize=7.2, fontweight="bold", rotation=0, ha="right", va="center")
            else:
                ax.set_ylabel("")

            if r == len(eo_themes) - 1:
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
                ax.tick_params(axis="x", labelrotation=90, labelsize=5.3, pad=1)
                ax2.tick_params(axis="x", labelbottom=False)
            else:
                ax.tick_params(axis="x", labelbottom=False)
                ax2.tick_params(axis="x", labelbottom=False)

            ax.tick_params(axis="y", labelsize=5.2)
            ax.tick_params(axis="y", colors=BLUESKY_COLOR, labelsize=5.2)
            ax2.tick_params(axis="y", colors=EO_COLOR, labelsize=5.2)
            if c != 0:
                ax.tick_params(axis="y", labelleft=False)
            if c != len(bluesky_themes) - 1:
                ax2.tick_params(axis="y", labelright=False)

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax2.spines["top"].set_visible(False)
            ax2.spines["left"].set_visible(False)
            ax.spines["left"].set_color(BLUESKY_COLOR)
            ax2.spines["right"].set_color(EO_COLOR)

    fig.suptitle(
        "Corrected pair time series matrix: Bluesky themes (B1-B14) vs EO themes (E1-E19)\nSeparate y-scales in each cell: Bluesky left axis, EO right axis",
        fontsize=14,
        fontweight="bold",
        y=0.997,
    )
    fig.text(0.5, 0.004, "Date", ha="center", va="bottom", fontsize=11, fontweight="bold")
    fig.text(0.003, 0.5, "EO themes (E1-E19)", ha="left", va="center", rotation=90, fontsize=11, fontweight="bold")

    out_base = FIG_DIR / "corrected_overlay_matrix_bluesky_columns_eo_rows"
    fig.savefig(out_base.with_suffix(".png"), dpi=FIG_DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)

def main() -> None:
    ensure_dirs()
    panel = load_panel()
    write_abbreviation_tables()
    build_matrix_figure(panel)

if __name__ == "__main__":
    main()
