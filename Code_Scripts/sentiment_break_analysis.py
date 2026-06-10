from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb
import pandas as pd
import ruptures as rpt

BASE_DIR = Path("local_data/03_Github_data/05_bluesky_data_merged_sentiment")
DATA_DIR = BASE_DIR / "data_"
OUTPUT_DIR = BASE_DIR / "theme_sentiment_break_analysis"
TABLE_DIR = OUTPUT_DIR / "tables"
PARQUET_GLOB = str(DATA_DIR / "*.parquet")

def export_csv(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    result = con.execute(query)
    headers = [d[0] for d in result.description]
    rows = result.fetchall()
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def queries() -> dict[str, str]:
    return {
        "sentiment_by_month": f"""
            WITH base AS (
              SELECT
                date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month,
                "record.text.clean.sentimentlabel" AS sentiment_label
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
            )
            SELECT month, sentiment_label, count(*) AS posts
            FROM base
            GROUP BY 1, 2
            ORDER BY 1, 2
        """,
        "sentiment_by_month_share": f"""
            WITH monthly AS (
              SELECT
                date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month,
                "record.text.clean.sentimentlabel" AS sentiment_label,
                count(*) AS posts
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
              GROUP BY 1, 2
            )
            SELECT
              month,
              sentiment_label,
              posts,
              round(100.0 * posts / sum(posts) OVER (PARTITION BY month), 4) AS pct_posts
            FROM monthly
            ORDER BY 1, 2
        """,
        "theme_sentiment_distribution": f"""
            SELECT
              "Theme_Name" AS theme_name,
              "record.text.clean.sentimentlabel" AS sentiment_label,
              count(*) AS posts
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY theme_name, posts DESC
        """,
        "theme_sentiment_share": f"""
            WITH theme_counts AS (
              SELECT
                "Theme_Name" AS theme_name,
                "record.text.clean.sentimentlabel" AS sentiment_label,
                count(*) AS posts
              FROM read_parquet('{PARQUET_GLOB}')
              GROUP BY 1, 2
            )
            SELECT
              theme_name,
              sentiment_label,
              posts,
              round(100.0 * posts / sum(posts) OVER (PARTITION BY theme_name), 4) AS pct_posts
            FROM theme_counts
            ORDER BY theme_name, posts DESC
        """,
        "theme_monthly_sentiment": f"""
            WITH base AS (
              SELECT
                date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month,
                "Theme_Name" AS theme_name,
                "record.text.clean.sentimentlabel" AS sentiment_label
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
            )
            SELECT month, theme_name, sentiment_label, count(*) AS posts
            FROM base
            GROUP BY 1, 2, 3
            ORDER BY 1, 2, 3
        """,
        "theme_monthly_totals": f"""
            WITH base AS (
              SELECT
                date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month,
                "Theme_Name" AS theme_name,
                "record.text.clean.sentimentlabel" AS sentiment_label
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
            ),
            monthly AS (
              SELECT month, theme_name, sentiment_label, count(*) AS posts
              FROM base
              GROUP BY 1, 2, 3
            )
            SELECT
              month,
              theme_name,
              sum(posts) AS total_posts,
              sum(CASE WHEN sentiment_label = 'negative' THEN posts ELSE 0 END) AS negative_posts,
              sum(CASE WHEN sentiment_label = 'neutral' THEN posts ELSE 0 END) AS neutral_posts,
              sum(CASE WHEN sentiment_label = 'positive' THEN posts ELSE 0 END) AS positive_posts,
              round(100.0 * sum(CASE WHEN sentiment_label = 'negative' THEN posts ELSE 0 END) / sum(posts), 4) AS negative_share_pct,
              round(100.0 * sum(CASE WHEN sentiment_label = 'neutral' THEN posts ELSE 0 END) / sum(posts), 4) AS neutral_share_pct,
              round(100.0 * sum(CASE WHEN sentiment_label = 'positive' THEN posts ELSE 0 END) / sum(posts), 4) AS positive_share_pct
            FROM monthly
            GROUP BY 1, 2
            ORDER BY 1, 2
        """,
        "top5_themes": f"""
            SELECT
              "Theme_Name" AS theme_name,
              count(*) AS posts
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY posts DESC
            LIMIT 5
        """,
    }

def compute_breaks() -> pd.DataFrame:
    monthly = pd.read_csv(TABLE_DIR / "theme_monthly_totals.csv")
    monthly["month"] = pd.to_datetime(monthly["month"], utc=True).dt.tz_convert(None)
    records: list[dict[str, object]] = []
    for theme, sub in monthly.groupby("theme_name"):
        sub = sub.sort_values("month").reset_index(drop=True).copy()
        if len(sub) < 7:
            continue
        total_std = sub["total_posts"].std(ddof=0) or 1.0
        neg_std = sub["negative_share_pct"].std(ddof=0) or 1.0
        signal = pd.DataFrame(
            {
                "volume_z": (sub["total_posts"] - sub["total_posts"].mean()) / total_std,
                "negshare_z": (sub["negative_share_pct"] - sub["negative_share_pct"].mean()) / neg_std,
            }
        ).to_numpy()
        try:
            algo = rpt.Binseg(model="l2").fit(signal)
            bkps = algo.predict(n_bkps=1)
        except Exception:
            bkps = [len(sub)]
        best_idx = int(bkps[0]) if bkps and 3 <= int(bkps[0]) <= len(sub) - 3 else None
        if best_idx is None or best_idx >= len(sub):
            continue
        pre = sub.iloc[best_idx - 3 : best_idx]
        post = sub.iloc[best_idx : best_idx + 3]
        volume_shift = abs(post["total_posts"].mean() - pre["total_posts"].mean()) / total_std
        neg_shift = abs(post["negative_share_pct"].mean() - pre["negative_share_pct"].mean()) / neg_std
        best_score = float(volume_shift + neg_shift)
        records.append(
            {
                "theme_name": theme,
                "break_month": sub.iloc[best_idx]["month"],
                "break_index": best_idx,
                "break_score": round(best_score, 4),
                "pre_total_posts_avg": round(pre["total_posts"].mean(), 2),
                "post_total_posts_avg": round(post["total_posts"].mean(), 2),
                "pre_negative_share_avg": round(pre["negative_share_pct"].mean(), 4),
                "post_negative_share_avg": round(post["negative_share_pct"].mean(), 4),
            }
        )
    out = pd.DataFrame(records).sort_values("break_score", ascending=False)
    out.to_csv(TABLE_DIR / "theme_structural_breaks.csv", index=False)
    return out

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    for name, query in queries().items():
        export_csv(con, query, TABLE_DIR / f"{name}.csv")
    compute_breaks()
    print(f"tables\t{TABLE_DIR}")

if __name__ == "__main__":
    main()
