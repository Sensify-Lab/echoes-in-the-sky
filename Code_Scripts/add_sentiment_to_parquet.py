from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from nltk.sentiment import SentimentIntensityAnalyzer


BASE_DIR = Path("local_data/03_Github_data/04_final_merged_10M")
OUTPUT_DIR = BASE_DIR / "add_sentiment"
TEXT_CLEAN_COL = "record.text.clean"
MEDIA_CLEAN_COL = "record.media.merge.clean"
BATCH_SIZE = 20_000
SAMPLE_ROWS_PER_FILE = 25
def sentiment_for_texts(texts: list[str | None]) -> tuple[list[float], list[str]]:
    sia = SentimentIntensityAnalyzer()
    scores: list[float] = []
    labels: list[str] = []
    for text in texts:
        if text is None:
            text_value = ""
        elif isinstance(text, str):
            text_value = text
        else:
            text_value = str(text)
        compound = float(sia.polarity_scores(text_value)["compound"])
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        scores.append(compound)
        labels.append(label)
    return scores, labels


def process_file(path_str: str) -> dict[str, str | int | float | list[list[str]]]:
    path = Path(path_str)
    out_path = OUTPUT_DIR / path.name.replace(".parquet", "_dual_sentiment.parquet")
    tmp_out_path = OUTPUT_DIR / path.name.replace(".parquet", "_dual_sentiment.tmp.parquet")
    parquet_file = pq.ParquetFile(path)
    schema_names = parquet_file.schema_arrow.names
    if TEXT_CLEAN_COL not in schema_names or MEDIA_CLEAN_COL not in schema_names:
        raise KeyError(f"Required columns missing in {path.name}")
    text_idx = schema_names.index(TEXT_CLEAN_COL)
    media_idx = schema_names.index(MEDIA_CLEAN_COL)

    writer: pq.ParquetWriter | None = None
    rows_written = 0
    sample_rows: list[list[str]] = []
    started = time.time()

    if tmp_out_path.exists():
        tmp_out_path.unlink()

    try:
        for batch in parquet_file.iter_batches(batch_size=BATCH_SIZE):
            text_clean_values = batch.column(text_idx).to_pylist()
            media_clean_values = batch.column(media_idx).to_pylist()
            text_scores, text_labels = sentiment_for_texts(text_clean_values)
            media_scores, media_labels = sentiment_for_texts(media_clean_values)
            out_batch = pa.record_batch(
                list(batch.columns)
                + [
                    pa.array(text_scores, type=pa.float32()),
                    pa.array(text_labels, type=pa.string()),
                    pa.array(media_scores, type=pa.float32()),
                    pa.array(media_labels, type=pa.string()),
                ],
                names=batch.schema.names
                + [
                    "record.text.clean.sentimentScore",
                    "record.text.clean.sentimentlabel",
                    "record.media.merge.clean.sentimentScore",
                    "record.media.merge.clean.sentimentlabel",
                ],
            )

            if writer is None:
                writer = pq.ParquetWriter(tmp_out_path, out_batch.schema, compression="snappy")
            writer.write_batch(out_batch)
            rows_written += out_batch.num_rows

            if len(sample_rows) < SAMPLE_ROWS_PER_FILE:
                take_n = min(SAMPLE_ROWS_PER_FILE - len(sample_rows), out_batch.num_rows)
                sample_table = pa.Table.from_batches([out_batch]).slice(0, take_n)
                sample_dict = sample_table.to_pydict()
                for i in range(take_n):
                    sample_rows.append(
                        [
                            path.name,
                            str(sample_dict.get("cid", [""])[i]),
                            str(sample_dict.get("author.handle", [""])[i]),
                            str(sample_dict.get(TEXT_CLEAN_COL, [""])[i]),
                            str(sample_dict.get("record.text.clean.sentimentScore", [""])[i]),
                            str(sample_dict.get("record.text.clean.sentimentlabel", [""])[i]),
                            str(sample_dict.get(MEDIA_CLEAN_COL, [""])[i]),
                            str(sample_dict.get("record.media.merge.clean.sentimentScore", [""])[i]),
                            str(sample_dict.get("record.media.merge.clean.sentimentlabel", [""])[i]),
                        ]
                    )
    finally:
        if writer is not None:
            writer.close()

    tmp_out_path.replace(out_path)

    elapsed = time.time() - started
    return {
        "file": path.name,
        "output_file": out_path.name,
        "rows_written": rows_written,
        "seconds": round(elapsed, 2),
        "sample_rows": sample_rows,
    }


def write_sample_csv(sample_rows: list[list[str]]) -> None:
    sample_path = OUTPUT_DIR / "sample_100_rows.csv"
    with sample_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source_file",
                "cid",
                "author.handle",
                "record.text.clean",
                "record.text.clean.sentimentScore",
                "record.text.clean.sentimentlabel",
                "record.media.merge.clean",
                "record.media.merge.clean.sentimentScore",
                "record.media.merge.clean.sentimentlabel",
            ]
        )
        writer.writerows(sample_rows[:100])


def write_summary(results: list[dict[str, str | int | float | list[list[str]]]]) -> None:
    summary_path = OUTPUT_DIR / "sentiment_processing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "output_file", "rows_written", "seconds"])
        for result in results:
            writer.writerow(
                [
                    result["file"],
                    result["output_file"],
                    result["rows_written"],
                    result["seconds"],
                ]
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cli_files = [Path(arg) for arg in sys.argv[1:]]
    files = cli_files if cli_files else sorted(BASE_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {BASE_DIR}")

    results: list[dict[str, str | int | float | list[list[str]]]] = []
    combined_sample_rows: list[list[str]] = []

    for path in files:
        result = process_file(str(path))
        results.append(result)
        combined_sample_rows.extend(result["sample_rows"])  # type: ignore[arg-type]
        print(
            f"processed {result['file']} -> {result['output_file']} "
            f"rows={result['rows_written']} seconds={result['seconds']}"
        )

    if not cli_files:
        results.sort(key=lambda x: str(x["file"]))
        write_sample_csv(combined_sample_rows)
        write_summary(results)
        print(f"saved sample csv to {OUTPUT_DIR / 'sample_100_rows.csv'}")


if __name__ == "__main__":
    main()
