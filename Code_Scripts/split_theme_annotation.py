from pathlib import Path

import pandas as pd


SOURCE_CSV = Path(
    "local_data/03_Github_data/01_Bluesky_Dataset_and_Analysis/ThemeGeneration_Step3_Annotation.csv"
)
OUTPUT_DIR = Path(
    "local_data/03_Github_data/02_microtopics_themes_annotation_seperate_May18"
)
PARTITION_COLUMN = "Data_Partition"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(SOURCE_CSV)
    partitions = sorted(df[PARTITION_COLUMN].dropna().unique().tolist())
    print(f"Found partitions: {partitions}")

    for part, subset in df.groupby(PARTITION_COLUMN, dropna=False):
        if pd.isna(part):
            output_name = "missing_theme_annotation.csv"
        else:
            output_name = f"{part}_theme_annotation.csv"

        output_path = OUTPUT_DIR / output_name
        subset.to_csv(output_path, index=False)
        print(f"Saved {output_name} with {len(subset)} rows")


if __name__ == "__main__":
    main()
