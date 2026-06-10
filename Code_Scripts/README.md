# Scripts Overview

The scripts in this folder are grouped by research task rather than by a shared application architecture. Most of them were written to run against local research data stored outside this repository under `local_data/...`.

## Main Groups

- `bluesky_data_collection.py`
  Collects Bluesky posts for a fixed query set and date window.

- `duckdb_analysis.py`, `analysis_queries.sql`, `publication_figures.py`, `selected_figures.py`
  Generate summary tables and publication-oriented figures from parquet inputs.

- `sentiment_*.py`, `add_sentiment_to_parquet.py`
  Sentiment labeling, monthly summaries, mismatch checks, break analysis, and sentiment figures.

- `eo_bluesky_*.py`
  Compare executive-order themes with Bluesky theme activity.

- `granger_*.py`
  Produce Granger-causality inputs, stationarity checks, heatmaps, and pairwise plots.

- `structure_break_*.py`
  Run structural-break analyses on theme activity over time.

## Important Assumptions

- Paths such as `local_data/03_Github_data/...` refer to the original research workspace and are not included in this repository.
- Most scripts write outputs to sibling directories near the input data.
- Several scripts expect intermediate CSV files produced by earlier steps rather than raw inputs only.

## Recommended Use

If you are adapting this repository:

1. Start by reading the constants at the top of each script.
2. Replace the hardcoded `BASE_DIR`, `WORK_DIR`, or parquet glob paths with your own local paths.
3. Run one analysis family at a time rather than trying to execute the whole folder as a pipeline.
4. Treat the checked-in sample CSVs as examples, not as complete reproduction inputs.
