from pathlib import Path

import pandas as pd


DATA_DIR = Path("local_data/03_Github_data/03_merged_data_annotation_may19")
DROP_COLUMNS = [
    "Generated_Initial_Codes",
    "Number_of_Posts",
    "Topic_Summarization",
    "Top_Keywords",
    "cluster",
]


def main() -> None:
    parquet_files = sorted(DATA_DIR.glob("*.parquet"))

    if not parquet_files:
        print(f"No parquet files found in: {DATA_DIR}")
        return

    for path in parquet_files:
        df = pd.read_parquet(path)
        existing_drop_columns = [col for col in DROP_COLUMNS if col in df.columns]
        missing_drop_columns = [col for col in DROP_COLUMNS if col not in df.columns]

        original_columns = list(df.columns)
        df = df.drop(columns=existing_drop_columns)
        df.to_parquet(path, index=False)

        print(f"Updated: {path.name}")
        print(f"  Original column count: {len(original_columns)}")
        print(f"  Removed columns: {existing_drop_columns}")
        print(f"  Missing requested columns: {missing_drop_columns}")
        print(f"  New column count: {len(df.columns)}")


if __name__ == "__main__":
    main()
