-- Reusable DuckDB queries for 03_Github_data/04_final_merged_10M
-- Update the parquet glob if you move the data folder.

WITH base AS (
  SELECT *, try_cast("record.created_at" AS TIMESTAMPTZ) AS record_created_ts
  FROM read_parquet('local_data/03_Github_data/04_final_merged_10M/*.parquet', filename=true)
)
SELECT count(*) AS total_rows FROM base;

SELECT regexp_extract(filename, '[^/]+$') AS file, count(*) AS rows
FROM read_parquet('local_data/03_Github_data/04_final_merged_10M/*.parquet', filename=true)
GROUP BY 1
ORDER BY 1;

SELECT date_trunc('month', try_cast("record.created_at" AS TIMESTAMPTZ)) AS month, count(*) AS rows
FROM read_parquet('local_data/03_Github_data/04_final_merged_10M/*.parquet')
WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
GROUP BY 1
ORDER BY 1;

SELECT "Theme_Name" AS theme_name, count(*) AS rows, round(avg("Theme_Confidence_Score"), 4) AS avg_confidence
FROM read_parquet('local_data/03_Github_data/04_final_merged_10M/*.parquet')
GROUP BY 1
ORDER BY rows DESC;
