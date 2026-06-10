from __future__ import annotations

import csv
import json
from pathlib import Path

import duckdb

BASE_DIR = Path("local_data/03_Github_data/05_bluesky_data_merged_sentiment")
DATA_DIR = BASE_DIR / "data_"
OUTPUT_DIR = BASE_DIR / "text_sentiment_analysis_outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
PARQUET_GLOB = str(DATA_DIR / "*.parquet")

def export_csv(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    result = con.execute(query)
    headers = [desc[0] for desc in result.description]
    rows = result.fetchall()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def q() -> dict[str, str]:
    return {
        "overview_metrics": f"""
            WITH base AS (
              SELECT *,
                     try_cast("record.created_at" AS TIMESTAMPTZ) AS record_created_ts
              FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT
              count(*) AS total_rows,
              count(DISTINCT cid) AS distinct_cid,
              count(DISTINCT uri) AS distinct_uri,
              count(DISTINCT "author.did") AS distinct_authors,
              count(DISTINCT "Theme_Name") AS distinct_themes,
              min(record_created_ts) AS min_record_created_at,
              max(record_created_ts) AS max_record_created_at,
              min(indexed_at) AS min_indexed_at,
              max(indexed_at) AS max_indexed_at,
              avg("record.text.clean.sentimentScore") AS avg_text_sentiment_score
            FROM base
        """,
        "sentiment_label_distribution": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              count(*) AS rows,
              round(100.0 * count(*) / sum(count(*)) OVER (), 4) AS pct_rows,
              round(avg("record.text.clean.sentimentScore"), 4) AS avg_sentiment_score
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY rows DESC
        """,
        "sentiment_score_summary": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              round(min("record.text.clean.sentimentScore"), 4) AS min_score,
              round(avg("record.text.clean.sentimentScore"), 4) AS avg_score,
              round(max("record.text.clean.sentimentScore"), 4) AS max_score,
              approx_quantile("record.text.clean.sentimentScore", [0.1, 0.25, 0.5, 0.75, 0.9]) AS score_quantiles
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY avg_score
        """,
        "theme_sentiment_distribution": f"""
            SELECT
              "Theme_Name" AS theme_name,
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              count(*) AS rows,
              round(avg("record.text.clean.sentimentScore"), 4) AS avg_score
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY theme_name, rows DESC
        """,
        "theme_sentiment_share": f"""
            WITH theme_counts AS (
              SELECT
                "Theme_Name" AS theme_name,
                "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
                count(*) AS rows
              FROM read_parquet('{PARQUET_GLOB}')
              GROUP BY 1, 2
            )
            SELECT
              theme_name,
              text_clean_sentimentlabel,
              rows,
              round(100.0 * rows / sum(rows) OVER (PARTITION BY theme_name), 4) AS pct_within_theme
            FROM theme_counts
            ORDER BY theme_name, rows DESC
        """,
        "top_themes_by_sentiment": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              "Theme_Name" AS theme_name,
              count(*) AS rows
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            QUALIFY row_number() OVER (PARTITION BY "record.text.clean.sentimentlabel" ORDER BY count(*) DESC) <= 10
            ORDER BY text_clean_sentimentlabel, rows DESC
        """,
        "monthly_sentiment_counts": f"""
            WITH base AS (
              SELECT
                date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month,
                "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
            )
            SELECT month, text_clean_sentimentlabel, count(*) AS rows
            FROM base
            GROUP BY 1, 2
            ORDER BY 1, 3 DESC
        """,
        "monthly_sentiment_share": f"""
            WITH monthly AS (
              SELECT
                date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month,
                "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
                count(*) AS rows
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
              GROUP BY 1, 2
            )
            SELECT
              month,
              text_clean_sentimentlabel,
              rows,
              round(100.0 * rows / sum(rows) OVER (PARTITION BY month), 4) AS pct_within_month
            FROM monthly
            ORDER BY 1, 2
        """,
        "quarterly_sentiment_counts": f"""
            WITH base AS (
              SELECT
                date_trunc('quarter', try_cast("record.created_at" AS TIMESTAMPTZ)) AS quarter,
                "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel
              FROM read_parquet('{PARQUET_GLOB}')
              WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
            )
            SELECT quarter, text_clean_sentimentlabel, count(*) AS rows
            FROM base
            GROUP BY 1, 2
            ORDER BY 1, 3 DESC
        """,
        "partition_sentiment_distribution": f"""
            SELECT
              "Data_Partition" AS data_partition,
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              count(*) AS rows
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY data_partition, rows DESC
        """,
        "engagement_by_sentiment": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              count(*) AS rows,
              round(avg(like_count), 4) AS avg_likes,
              round(avg(reply_count), 4) AS avg_replies,
              round(avg(repost_count), 4) AS avg_reposts,
              round(avg(quote_count), 4) AS avg_quotes,
              approx_quantile(like_count, [0.5, 0.9, 0.99]) AS like_quantiles,
              approx_quantile(reply_count, [0.5, 0.9, 0.99]) AS reply_quantiles
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY rows DESC
        """,
        "top_authors_by_sentiment": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              "author.handle" AS author_handle,
              count(*) AS posts,
              sum(like_count + reply_count + repost_count + quote_count) AS total_engagement
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            QUALIFY row_number() OVER (
              PARTITION BY "record.text.clean.sentimentlabel"
              ORDER BY count(*) DESC, sum(like_count + reply_count + repost_count + quote_count) DESC
            ) <= 25
            ORDER BY text_clean_sentimentlabel, posts DESC
        """,
        "content_features_by_sentiment": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              count(*) AS rows,
              sum("record.embed.external.uri" IS NOT NULL) AS external_links,
              sum(array_length("record.embed.images") > 0) AS image_embeds,
              sum(array_length("record.embed.media.images") > 0) AS media_image_embeds,
              sum("record.embed.video.ref.link" IS NOT NULL) AS video_embeds,
              sum(array_length("record.hashtag") > 0) AS hashtagged_posts,
              round(avg("record.text.clean.count"), 2) AS avg_clean_text_count
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY rows DESC
        """,
        "language_sentiment_top": f"""
            WITH langs AS (
              SELECT
                "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
                unnest("record.langs") AS lang
              FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT
              text_clean_sentimentlabel,
              coalesce(lang, 'NULL') AS lang,
              count(*) AS rows
            FROM langs
            GROUP BY 1, 2
            QUALIFY row_number() OVER (PARTITION BY text_clean_sentimentlabel ORDER BY count(*) DESC) <= 15
            ORDER BY text_clean_sentimentlabel, rows DESC
        """,
        "media_label_match_by_text_label": f"""
            SELECT
              "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
              "record.media.merge.clean.sentimentlabel" AS media_merge_clean_sentimentlabel,
              count(*) AS rows
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY text_clean_sentimentlabel, rows DESC
        """,
        "theme_mismatch_counts": f"""
            SELECT
              "Theme_Name" AS theme_name,
              count(*) AS rows,
              sum(
                CASE
                  WHEN "record.text.clean.sentimentlabel"
                       IS DISTINCT FROM
                       "record.media.merge.clean.sentimentlabel"
                  THEN 1 ELSE 0
                END
              ) AS mismatch_rows,
              round(
                100.0 * sum(
                  CASE
                    WHEN "record.text.clean.sentimentlabel"
                         IS DISTINCT FROM
                         "record.media.merge.clean.sentimentlabel"
                    THEN 1 ELSE 0
                  END
                ) / count(*),
                4
              ) AS mismatch_pct
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY mismatch_rows DESC
        """,
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    queries = q()
    for name, query in queries.items():
        export_csv(con, query, TABLE_DIR / f"{name}.csv")

    print(f"saved_tables\t{TABLE_DIR}")

if __name__ == "__main__":
    main()
