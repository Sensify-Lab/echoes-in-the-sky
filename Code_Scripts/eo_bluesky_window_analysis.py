from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

BASE_DIR = Path("local_data/03_Github_data/07_EO_effect_bluesky_2025_01_20_to_2026_02_01")
TABLE_DIR = BASE_DIR / "tables"
BASE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_START = "2025-01-20"
WINDOW_END = "2026-02-01"

BLUESKY_GLOB = "local_data/03_Github_data/04_final_merged_10M/*.parquet"
EO_CSV = "local_data/03_Github_data/00_Executive_Orders_Dataset_and_Analysis/EO_ThemeGeneration_Step3_Annotation.csv"
CROSSWALK_CSV = BASE_DIR / "eo_bluesky_theme_crosswalk.csv"

CROSSWALK_ROWS = [
    ("Executive Coordination and Administrative Task Forces", "Executive Power, Patronage, and Administrative Control", "high"),
    ("Public Order, Crime, and Border Enforcement", "Immigration, Borders, and Sovereignty", "medium"),
    ("Foreign Policy and External Relations", "National Security, Military, and Geopolitical Conflict", "high"),
    ("National Security, Defense, and Strategic Resilience", "National Security, Military, and Geopolitical Conflict", "high"),
    ("Trade, Tariffs, and Export Promotion", "Economic Policy, Taxation, and Distributional Conflict", "high"),
    ("Domestic Investment and Market Competition", "Economic Policy, Taxation, and Distributional Conflict", "medium"),
    ("Civil Rights, Equality, and Gender Policy", "Race, Civil Rights, and Identity Hierarchies", "medium"),
    ("Health, Reproductive Care, and Medical Research", "Health, Public Health, and Bodily Politics", "high"),
    ("Energy Development and Resource Expansion", "Public Lands, Environment, and Climate Governance", "medium"),
    ("Education and Higher-Ed Transparency", "Education, Universities, and Intellectual Control", "high"),
    ("Technology, Science, and Artificial Intelligence Development", "Technology, Platforms, and Digital Power", "high"),
    ("Workforce, Training, and Retirement Security", "Economic Policy, Taxation, and Distributional Conflict", "medium"),
    ("Cultural Heritage and National Pastime", "Aesthetics, Symbolism, and Performance of Power", "medium"),
    ("Environmental Stewardship and Beautification", "Public Lands, Environment, and Climate Governance", "high"),
    ("Housing Affordability and Development Deregulation", "Economic Policy, Taxation, and Distributional Conflict", "medium"),
    ("Infrastructure and Maritime Transportation Rules", "Economic Policy, Taxation, and Distributional Conflict", "low"),
    ("Public Lands, Parks, and Timber Production", "Public Lands, Environment, and Climate Governance", "high"),
    ("Water Supply and Disaster Response", "Public Lands, Environment, and Climate Governance", "low"),
    ("Family Support and Child Wellbeing", "Health, Public Health, and Bodily Politics", "low"),
]

def export_csv(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    result = con.execute(query)
    headers = [d[0] for d in result.description]
    rows = result.fetchall()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def write_crosswalk() -> None:
    with CROSSWALK_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["eo_theme", "bluesky_theme", "mapping_confidence"])
        writer.writerows(CROSSWALK_ROWS)

def build_core_tables(con: duckdb.DuckDBPyConnection) -> None:
    bs_where = f"""
        try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
        AND CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE) >= DATE '{WINDOW_START}'
        AND CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE) <= DATE '{WINDOW_END}'
    """
    eo_where = f"""
        CAST(signing_date AS DATE) >= DATE '{WINDOW_START}'
        AND CAST(signing_date AS DATE) <= DATE '{WINDOW_END}'
    """
    queries = {
        "window_summary": f"""
            SELECT
              DATE '{WINDOW_START}' AS window_start,
              DATE '{WINDOW_END}' AS window_end,
              (SELECT count(*) FROM read_parquet('{BLUESKY_GLOB}') WHERE {bs_where}) AS bluesky_rows,
              (SELECT count(*) FROM read_csv_auto('{EO_CSV}', header=true) WHERE {eo_where}) AS eo_rows
        """,
        "crosswalk_inventory": f"""
            SELECT * FROM read_csv_auto('{CROSSWALK_CSV}', header=true) ORDER BY eo_theme
        """,
        "eo_theme_counts": f"""
            SELECT
              eo.Finalized_Theme_Name AS eo_theme,
              cw.bluesky_theme,
              cw.mapping_confidence,
              count(*) AS eo_count
            FROM read_csv_auto('{EO_CSV}', header=true) eo
            LEFT JOIN read_csv_auto('{CROSSWALK_CSV}', header=true) cw
              ON eo.Finalized_Theme_Name = cw.eo_theme
            WHERE {eo_where}
            GROUP BY 1, 2, 3
            ORDER BY eo_count DESC
        """,
        "bluesky_theme_counts": f"""
            SELECT
              Theme_Name AS bluesky_theme,
              count(*) AS bluesky_posts,
              count(DISTINCT "author.did") AS distinct_authors
            FROM read_parquet('{BLUESKY_GLOB}')
            WHERE {bs_where}
            GROUP BY 1
            ORDER BY bluesky_posts DESC
        """,
        "eo_daily_overall": f"""
            SELECT CAST(signing_date AS DATE) AS day, count(*) AS eo_count
            FROM read_csv_auto('{EO_CSV}', header=true)
            WHERE {eo_where}
            GROUP BY 1
            ORDER BY 1
        """,
        "bluesky_daily_overall": f"""
            SELECT CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE) AS day, count(*) AS bluesky_posts
            FROM read_parquet('{BLUESKY_GLOB}')
            WHERE {bs_where}
            GROUP BY 1
            ORDER BY 1
        """,
        "eo_monthly_overall": f"""
            SELECT date_trunc('month', CAST(signing_date AS DATE)) AS month, count(*) AS eo_count
            FROM read_csv_auto('{EO_CSV}', header=true)
            WHERE {eo_where}
            GROUP BY 1
            ORDER BY 1
        """,
        "bluesky_monthly_overall": f"""
            SELECT date_trunc('month', CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE)) AS month, count(*) AS bluesky_posts
            FROM read_parquet('{BLUESKY_GLOB}')
            WHERE {bs_where}
            GROUP BY 1
            ORDER BY 1
        """,
        "eo_theme_daily": f"""
            SELECT
              CAST(eo.signing_date AS DATE) AS day,
              eo.Finalized_Theme_Name AS eo_theme,
              cw.bluesky_theme,
              count(*) AS eo_count
            FROM read_csv_auto('{EO_CSV}', header=true) eo
            LEFT JOIN read_csv_auto('{CROSSWALK_CSV}', header=true) cw
              ON eo.Finalized_Theme_Name = cw.eo_theme
            WHERE {eo_where}
            GROUP BY 1, 2, 3
            ORDER BY 1, 4 DESC
        """,
        "bluesky_theme_daily": f"""
            SELECT
              CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE) AS day,
              Theme_Name AS bluesky_theme,
              count(*) AS bluesky_posts
            FROM read_parquet('{BLUESKY_GLOB}')
            WHERE {bs_where}
            GROUP BY 1, 2
            ORDER BY 1, 3 DESC
        """,
        "eo_theme_monthly": f"""
            SELECT
              date_trunc('month', CAST(eo.signing_date AS DATE)) AS month,
              eo.Finalized_Theme_Name AS eo_theme,
              cw.bluesky_theme,
              count(*) AS eo_count
            FROM read_csv_auto('{EO_CSV}', header=true) eo
            LEFT JOIN read_csv_auto('{CROSSWALK_CSV}', header=true) cw
              ON eo.Finalized_Theme_Name = cw.eo_theme
            WHERE {eo_where}
            GROUP BY 1, 2, 3
            ORDER BY 1, 4 DESC
        """,
        "bluesky_theme_monthly": f"""
            SELECT
              date_trunc('month', CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE)) AS month,
              Theme_Name AS bluesky_theme,
              count(*) AS bluesky_posts
            FROM read_parquet('{BLUESKY_GLOB}')
            WHERE {bs_where}
            GROUP BY 1, 2
            ORDER BY 1, 3 DESC
        """,
    }
    for name, query in queries.items():
        export_csv(con, query, TABLE_DIR / f"{name}.csv")

def build_event_studies() -> None:
    eo_daily = pd.read_csv(TABLE_DIR / "eo_theme_daily.csv")
    bs_daily = pd.read_csv(TABLE_DIR / "bluesky_theme_daily.csv")
    eo_daily["day"] = pd.to_datetime(eo_daily["day"])
    bs_daily["day"] = pd.to_datetime(bs_daily["day"])
    eo_daily = eo_daily.dropna(subset=["bluesky_theme"]).copy()
    bs_lookup = bs_daily.set_index(["bluesky_theme", "day"])["bluesky_posts"].to_dict()
    eo_lookup = eo_daily.groupby(["bluesky_theme", "day"], as_index=False)["eo_count"].sum().set_index(["bluesky_theme", "day"])["eo_count"].to_dict()

    rows = []
    for _, row in eo_daily.iterrows():
        theme = row["bluesky_theme"]
        event_day = row["day"]
        for rel in range(-14, 15):
            target = event_day + pd.Timedelta(days=rel)
            rows.append({
                "theme": theme,
                "event_day": event_day,
                "relative_day": rel,
                "avg_bluesky_posts": bs_lookup.get((theme, target), 0),
                "eo_count": eo_lookup.get((theme, target), 0),
            })
    event_df = pd.DataFrame(rows)
    event_df = event_df.groupby(["theme", "relative_day"], as_index=False).agg(
        avg_bluesky_posts=("avg_bluesky_posts", "mean"),
        avg_eo_count=("eo_count", "mean"),
        num_events=("event_day", "nunique"),
    )
    event_df.to_csv(TABLE_DIR / "event_study_eo_to_bluesky.csv", index=False)

    spike_rows = []
    for theme, sub in bs_daily.groupby("bluesky_theme"):
        sub = sub.sort_values("day").copy()
        if len(sub) < 10:
            continue
        threshold = sub["bluesky_posts"].quantile(0.95)
        spikes = sub[sub["bluesky_posts"] >= threshold]
        for _, row in spikes.iterrows():
            event_day = row["day"]
            for rel in range(-14, 15):
                target = event_day + pd.Timedelta(days=rel)
                spike_rows.append({
                    "theme": theme,
                    "event_day": event_day,
                    "relative_day": rel,
                    "avg_eo_count": eo_lookup.get((theme, target), 0),
                    "avg_bluesky_posts": bs_lookup.get((theme, target), 0),
                })
    spike_df = pd.DataFrame(spike_rows)
    spike_df = spike_df.groupby(["theme", "relative_day"], as_index=False).agg(
        avg_eo_count=("avg_eo_count", "mean"),
        avg_bluesky_posts=("avg_bluesky_posts", "mean"),
        num_spikes=("event_day", "nunique"),
    )
    spike_df.to_csv(TABLE_DIR / "event_study_bluesky_spikes_to_eo.csv", index=False)

def build_crosscorr_and_granger() -> None:
    eo_daily = pd.read_csv(TABLE_DIR / "eo_theme_daily.csv")
    bs_daily = pd.read_csv(TABLE_DIR / "bluesky_theme_daily.csv")
    eo_daily["day"] = pd.to_datetime(eo_daily["day"])
    bs_daily["day"] = pd.to_datetime(bs_daily["day"])
    themes = sorted(set(eo_daily["bluesky_theme"].dropna()) & set(bs_daily["bluesky_theme"].dropna()))
    cc_rows = []
    granger_rows = []
    for theme in themes:
        eo_sub = eo_daily[eo_daily["bluesky_theme"] == theme].groupby("day", as_index=True)["eo_count"].sum()
        bs_sub = bs_daily[bs_daily["bluesky_theme"] == theme].groupby("day", as_index=True)["bluesky_posts"].sum()
        idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
        eo_series = eo_sub.reindex(idx, fill_value=0).astype(float)
        bs_series = bs_sub.reindex(idx, fill_value=0).astype(float)
        if eo_series.std() == 0 or bs_series.std() == 0:
            continue
        for lag in range(-21, 22):
            if lag < 0:
                corr = eo_series[:lag].corr(bs_series[-lag:])
            elif lag > 0:
                corr = eo_series[lag:].corr(bs_series[:-lag])
            else:
                corr = eo_series.corr(bs_series)
            cc_rows.append({"theme": theme, "lag_days": lag, "correlation": corr})

        week_idx = pd.date_range(WINDOW_START, WINDOW_END, freq="W")
        eo_week = eo_series.resample("W").sum().reindex(week_idx, fill_value=0)
        bs_week = bs_series.resample("W").sum().reindex(week_idx, fill_value=0)
        df = pd.DataFrame({"eo": eo_week, "bluesky": bs_week}).reset_index(drop=True)
        if len(df) >= 12 and df["eo"].sum() > 0 and df["bluesky"].std() > 0:
            try:
                res1 = grangercausalitytests(df[["bluesky", "eo"]], maxlag=3, verbose=False)
                p1 = min(res1[l][0]["ssr_ftest"][1] for l in res1)
                res2 = grangercausalitytests(df[["eo", "bluesky"]], maxlag=3, verbose=False)
                p2 = min(res2[l][0]["ssr_ftest"][1] for l in res2)
                granger_rows.append({"theme": theme, "eo_to_bluesky_min_p": p1, "bluesky_to_eo_min_p": p2, "weeks": len(df)})
            except Exception:
                continue

    cc = pd.DataFrame(cc_rows)
    cc.to_csv(TABLE_DIR / "theme_cross_correlations.csv", index=False)
    peak = cc.loc[cc.groupby("theme")["correlation"].apply(lambda s: s.abs().idxmax())].copy()
    peak.to_csv(TABLE_DIR / "theme_cross_correlation_peaks.csv", index=False)
    pd.DataFrame(granger_rows).sort_values("theme").to_csv(TABLE_DIR / "theme_granger_results.csv", index=False)

def main() -> None:
    write_crosswalk()
    con = duckdb.connect()
    build_core_tables(con)
    build_event_studies()
    build_crosscorr_and_granger()
    print(f"tables\t{TABLE_DIR}")

if __name__ == "__main__":
    main()
