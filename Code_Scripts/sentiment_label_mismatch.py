from __future__ import annotations

from pathlib import Path

import duckdb


BASE_DIR = Path("local_data/03_Github_data/05_bluesky_data_merged_sentiment")
DATA_DIR = BASE_DIR / "data_"
PARQUET_GLOB = str(DATA_DIR / "*.parquet")
OUTPUT_DIR = BASE_DIR / "analysis_outputs"
MISMATCH_CSV = OUTPUT_DIR / "sentiment_label_mismatches.csv"
SUMMARY_CSV = OUTPUT_DIR / "sentiment_label_mismatch_summary.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    mismatch_condition = """
        "record.text.clean.sentimentlabel"
        IS DISTINCT FROM
        "record.media.merge.clean.sentimentlabel"
    """

    summary_query = f"""
        SELECT
            count(*) AS total_rows,
            sum(CASE WHEN {mismatch_condition} THEN 1 ELSE 0 END) AS mismatched_rows,
            round(
                100.0 * sum(CASE WHEN {mismatch_condition} THEN 1 ELSE 0 END) / count(*),
                4
            ) AS mismatched_pct
        FROM read_parquet('{PARQUET_GLOB}')
    """

    mismatch_rows_query = f"""
        SELECT
            regexp_extract(filename, '[^/]+$') AS source_file,
            cid,
            uri,
            "author.handle" AS author_handle,
            "record.text.clean" AS record_text_clean,
            "record.media.merge.clean" AS record_media_merge_clean,
            "record.text.clean.sentimentlabel" AS text_clean_sentimentlabel,
            "record.media.merge.clean.sentimentlabel" AS media_merge_clean_sentimentlabel,
            "record.text.clean.sentimentScore" AS text_clean_sentimentScore,
            "record.media.merge.clean.sentimentScore" AS media_merge_clean_sentimentScore
        FROM read_parquet('{PARQUET_GLOB}', filename=true)
        WHERE {mismatch_condition}
    """

    con.execute(
        f"COPY ({summary_query}) TO '{SUMMARY_CSV.as_posix()}' (HEADER, DELIMITER ',')"
    )
    con.execute(
        f"COPY ({mismatch_rows_query}) TO '{MISMATCH_CSV.as_posix()}' (HEADER, DELIMITER ',')"
    )

    total_rows, mismatched_rows, mismatched_pct = con.execute(summary_query).fetchone()
    print(f"total_rows\t{total_rows}")
    print(f"mismatched_rows\t{mismatched_rows}")
    print(f"mismatched_pct\t{mismatched_pct}")
    print(f"summary_csv\t{SUMMARY_CSV}")
    print(f"mismatch_csv\t{MISMATCH_CSV}")


if __name__ == "__main__":
    main()
