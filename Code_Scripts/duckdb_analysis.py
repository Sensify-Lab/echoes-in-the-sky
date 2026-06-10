from __future__ import annotations

import json
import csv
from pathlib import Path

import duckdb

BASE_DIR = Path("local_data/03_Github_data/04_final_merged_10M")
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
PARQUET_GLOB = str(BASE_DIR / "*.parquet")

def qident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def export_csv(con: duckdb.DuckDBPyConnection, query: str, path: Path) -> None:
    result = con.execute(query)
    headers = [desc[0] for desc in result.description]
    rows = result.fetchall()
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")

def build_queries() -> dict[str, str]:
    rc = qident("record.created_at")
    rt = qident("record.text")
    rtc = qident("record.text.clean")
    ah = qident("author.handle")
    ad = qident("author.did")
    al = qident("author.labels")
    dp = qident("Data_Partition")
    th = qident("Theme_Name")
    tcs = qident("Theme_Confidence_Score")
    cl = qident("Cluster_ID")
    rel = qident("record.embed.external.uri")
    rei = qident("record.embed.images")
    remi = qident("record.embed.media.images")
    rvl = qident("record.embed.video.ref.link")
    rht = qident("record.hashtag")
    rcc = qident("record.text.clean.count")
    rlangs = qident("record.langs")
    ac = qident("author.created_at")

    from_clause = f"read_parquet('{PARQUET_GLOB}', filename=true)"

    return {
        "schema": f"DESCRIBE SELECT * FROM read_parquet('{PARQUET_GLOB}')",
        "file_row_counts": f"""
            SELECT regexp_extract(filename, '[^/]+$') AS file, count(*) AS rows
            FROM {from_clause}
            GROUP BY 1
            ORDER BY 1
        """,
        "overview_metrics": f"""
            WITH base AS (
              SELECT *, try_cast({rc} AS TIMESTAMPTZ) AS record_created_ts
              FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT
              count(*) AS total_rows,
              count(DISTINCT cid) AS distinct_cid,
              count(DISTINCT uri) AS distinct_uri,
              count(DISTINCT {ad}) AS distinct_authors,
              count(DISTINCT {cl}) AS distinct_clusters,
              count(DISTINCT {th}) AS distinct_themes,
              min(indexed_at) AS min_indexed_at,
              max(indexed_at) AS max_indexed_at,
              min(record_created_ts) AS min_record_created_at,
              max(record_created_ts) AS max_record_created_at,
              sum({rt} IS NULL) AS null_record_text,
              sum({rtc} IS NULL) AS null_clean_text,
              sum({ah} IS NULL) AS null_author_handle,
              sum(record_created_ts IS NULL) AS unparsable_record_created_at,
              round(avg(like_count), 4) AS avg_like_count,
              round(avg(reply_count), 4) AS avg_reply_count,
              round(avg(repost_count), 4) AS avg_repost_count,
              round(avg(quote_count), 4) AS avg_quote_count
            FROM base
        """,
        "data_quality_checks": f"""
            WITH counts AS (
              SELECT
                count(*) AS total_rows,
                count(DISTINCT cid) AS distinct_cid,
                count(DISTINCT uri) AS distinct_uri,
                count(DISTINCT concat(cid, '|', uri)) AS distinct_cid_uri_pairs
              FROM {from_clause}
            ),
            time_slices AS (
              SELECT
                sum(try_cast({rc} AS TIMESTAMPTZ) < TIMESTAMPTZ '2024-01-01') AS pre_2024_rows,
                sum(try_cast({rc} AS TIMESTAMPTZ) < TIMESTAMPTZ '2024-11-01') AS pre_nov_2024_rows,
                sum(try_cast({rc} AS TIMESTAMPTZ) >= TIMESTAMPTZ '2025-01-01'
                    AND try_cast({rc} AS TIMESTAMPTZ) < TIMESTAMPTZ '2026-01-01') AS rows_2025,
                sum(try_cast({rc} AS TIMESTAMPTZ) >= TIMESTAMPTZ '2026-01-01') AS rows_2026
              FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT
              total_rows,
              total_rows - distinct_cid AS duplicate_cid_rows,
              total_rows - distinct_uri AS duplicate_uri_rows,
              total_rows - distinct_cid_uri_pairs AS duplicate_cid_uri_pair_rows,
              pre_2024_rows,
              pre_nov_2024_rows,
              rows_2025,
              rows_2026
            FROM counts, time_slices
        """,
        "partition_distribution": f"""
            SELECT {dp} AS data_partition, count(*) AS rows
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY rows DESC
        """,
        "theme_distribution": f"""
            WITH total AS (
              SELECT count(*) AS n FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT
              {th} AS theme_name,
              count(*) AS rows,
              round(100.0 * count(*) / (SELECT n FROM total), 2) AS pct_rows,
              round(avg({tcs}), 4) AS avg_confidence
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY rows DESC
        """,
        "per_file_summary": f"""
            WITH base AS (
              SELECT
                regexp_extract(filename, '[^/]+$') AS file,
                try_cast({rc} AS TIMESTAMPTZ) AS record_created_ts,
                indexed_at,
                {th} AS theme_name,
                like_count,
                reply_count,
                repost_count,
                quote_count
              FROM {from_clause}
            )
            SELECT
              file,
              count(*) AS rows,
              min(record_created_ts) AS min_created,
              max(record_created_ts) AS max_created,
              min(indexed_at) AS min_indexed,
              max(indexed_at) AS max_indexed,
              count(DISTINCT theme_name) AS themes,
              round(avg(like_count), 2) AS avg_likes,
              round(avg(reply_count), 2) AS avg_replies,
              round(avg(repost_count), 2) AS avg_reposts,
              round(avg(quote_count), 2) AS avg_quotes
            FROM base
            GROUP BY 1
            ORDER BY 1
        """,
        "monthly_trends": f"""
            WITH base AS (
              SELECT try_cast({rc} AS TIMESTAMPTZ) AS record_created_ts
              FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT
              date_trunc('month', record_created_ts) AS month,
              count(*) AS rows
            FROM base
            WHERE record_created_ts IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """,
        "theme_monthly_top8": f"""
            WITH top_themes AS (
              SELECT {th} AS theme_name
              FROM read_parquet('{PARQUET_GLOB}')
              GROUP BY 1
              ORDER BY count(*) DESC
              LIMIT 8
            )
            SELECT
              date_trunc('month', try_cast({rc} AS TIMESTAMPTZ)) AS month,
              {th} AS theme_name,
              count(*) AS rows
            FROM read_parquet('{PARQUET_GLOB}')
            WHERE {th} IN (SELECT theme_name FROM top_themes)
              AND try_cast({rc} AS TIMESTAMPTZ) IS NOT NULL
            GROUP BY 1, 2
            ORDER BY 1, 3 DESC
        """,
        "top_languages": f"""
            WITH langs AS (
              SELECT unnest({rlangs}) AS lang
              FROM read_parquet('{PARQUET_GLOB}')
            )
            SELECT coalesce(lang, 'NULL') AS lang, count(*) AS uses
            FROM langs
            GROUP BY 1
            ORDER BY uses DESC
            LIMIT 50
        """,
        "content_shape_metrics": f"""
            SELECT
              sum({rel} IS NOT NULL) AS external_links,
              sum(array_length({rei}) > 0) AS image_embeds,
              sum(array_length({remi}) > 0) AS media_image_embeds,
              sum({rvl} IS NOT NULL) AS video_embeds,
              sum(array_length({rht}) > 0) AS hashtagged_posts,
              round(avg({rcc}), 2) AS avg_clean_text_tokens,
              approx_quantile({rcc}, [0.5, 0.9, 0.99]) AS clean_text_count_quantiles
            FROM read_parquet('{PARQUET_GLOB}')
        """,
        "engagement_distribution": f"""
            SELECT
              sum(like_count = 0) AS zero_likes,
              sum(reply_count = 0) AS zero_replies,
              sum(repost_count = 0) AS zero_reposts,
              sum(quote_count = 0) AS zero_quotes,
              approx_quantile(like_count, [0.5, 0.9, 0.99]) AS like_quantiles,
              approx_quantile(reply_count, [0.5, 0.9, 0.99]) AS reply_quantiles,
              approx_quantile(repost_count, [0.5, 0.9, 0.99]) AS repost_quantiles,
              approx_quantile(quote_count, [0.5, 0.9, 0.99]) AS quote_quantiles
            FROM read_parquet('{PARQUET_GLOB}')
        """,
        "top_authors_by_posts": f"""
            SELECT
              {ad} AS author_did,
              {ah} AS author_handle,
              count(*) AS posts,
              sum(like_count) AS total_likes,
              sum(reply_count) AS total_replies,
              sum(repost_count) AS total_reposts,
              sum(quote_count) AS total_quotes
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY posts DESC, total_likes DESC
            LIMIT 100
        """,
        "top_authors_by_total_engagement": f"""
            SELECT
              {ad} AS author_did,
              {ah} AS author_handle,
              count(*) AS posts,
              sum(like_count + reply_count + repost_count + quote_count) AS total_engagement,
              sum(like_count) AS total_likes,
              sum(reply_count) AS total_replies,
              sum(repost_count) AS total_reposts,
              sum(quote_count) AS total_quotes
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY total_engagement DESC, posts DESC
            LIMIT 100
        """,
        "top_clusters": f"""
            SELECT
              {cl} AS cluster_id,
              {th} AS theme_name,
              count(*) AS rows
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1, 2
            ORDER BY rows DESC
            LIMIT 100
        """,
        "theme_author_summary": f"""
            SELECT
              {th} AS theme_name,
              count(*) AS rows,
              count(DISTINCT {ad}) AS distinct_authors,
              round(avg(like_count), 2) AS avg_likes,
              round(avg(reply_count), 2) AS avg_replies,
              round(avg(repost_count), 2) AS avg_reposts,
              round(avg(quote_count), 2) AS avg_quotes
            FROM read_parquet('{PARQUET_GLOB}')
            GROUP BY 1
            ORDER BY rows DESC
        """,
        "author_account_age": f"""
            SELECT
              date_trunc('month', try_cast({ac} AS TIMESTAMPTZ)) AS author_created_month,
              count(*) AS posts
            FROM read_parquet('{PARQUET_GLOB}')
            WHERE try_cast({ac} AS TIMESTAMPTZ) IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """,
        "reusable_sql": f"""
-- Reusable DuckDB queries for 03_Github_data/04_final_merged_10M
-- Update the parquet glob if you move the data folder.

WITH base AS (
  SELECT *, try_cast({rc} AS TIMESTAMPTZ) AS record_created_ts
  FROM read_parquet('{PARQUET_GLOB}', filename=true)
)
SELECT count(*) AS total_rows FROM base;

SELECT regexp_extract(filename, '[^/]+$') AS file, count(*) AS rows
FROM read_parquet('{PARQUET_GLOB}', filename=true)
GROUP BY 1
ORDER BY 1;

SELECT date_trunc('month', try_cast({rc} AS TIMESTAMPTZ)) AS month, count(*) AS rows
FROM read_parquet('{PARQUET_GLOB}')
WHERE try_cast({rc} AS TIMESTAMPTZ) IS NOT NULL
GROUP BY 1
ORDER BY 1;

SELECT {th} AS theme_name, count(*) AS rows, round(avg({tcs}), 4) AS avg_confidence
FROM read_parquet('{PARQUET_GLOB}')
GROUP BY 1
ORDER BY rows DESC;
        """.strip()
    }

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    queries = build_queries()

    csv_exports = {
        "schema.csv": queries["schema"],
        "file_row_counts.csv": queries["file_row_counts"],
        "overview_metrics.csv": queries["overview_metrics"],
        "data_quality_checks.csv": queries["data_quality_checks"],
        "partition_distribution.csv": queries["partition_distribution"],
        "theme_distribution.csv": queries["theme_distribution"],
        "per_file_summary.csv": queries["per_file_summary"],
        "monthly_trends.csv": queries["monthly_trends"],
        "theme_monthly_top8.csv": queries["theme_monthly_top8"],
        "top_languages.csv": queries["top_languages"],
        "content_shape_metrics.csv": queries["content_shape_metrics"],
        "engagement_distribution.csv": queries["engagement_distribution"],
        "top_authors_by_posts.csv": queries["top_authors_by_posts"],
        "top_authors_by_total_engagement.csv": queries["top_authors_by_total_engagement"],
        "top_clusters.csv": queries["top_clusters"],
        "theme_author_summary.csv": queries["theme_author_summary"],
        "author_account_age.csv": queries["author_account_age"],
    }

    for file_name, query in csv_exports.items():
        export_csv(con, query, OUTPUT_DIR / file_name)

    write_text(OUTPUT_DIR / "analysis_queries.sql", queries["reusable_sql"] + "\n")
    print(f"Saved analysis outputs to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
