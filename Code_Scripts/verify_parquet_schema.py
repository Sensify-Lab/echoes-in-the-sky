from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_PATH = Path(
    "local_data/03_Github_data/05_bluesky_data_merged_sentiment/data_/part1_config_0001_merged_with_microtopics_input_2541294by47_dual_sentiment.parquet"
)


def flatten_field(field: pa.Field, prefix: str = "") -> list[str]:
    name = f"{prefix}.{field.name}" if prefix else field.name
    out = [name]
    field_type = field.type

    if pa.types.is_struct(field_type):
        for child in field_type:
            out.extend(flatten_field(child, name))
    elif pa.types.is_list(field_type) or pa.types.is_large_list(field_type):
        value_field = field_type.value_field
        out.append(f"{name}[]")
        if pa.types.is_struct(value_field.type):
            for child in value_field.type:
                out.extend(flatten_field(child, f"{name}[]"))
    return out


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow

    top_level = list(schema.names)
    expanded: list[str] = []
    for field in schema:
        expanded.extend(flatten_field(field))

    print(f"file\t{path}")
    print(f"rows\t{pf.metadata.num_rows}")
    print(f"top_level_columns\t{len(top_level)}")
    print(f"expanded_fields\t{len(expanded)}")
    print()

    print("TOP_LEVEL_COLUMNS")
    for i, name in enumerate(top_level, start=1):
        print(f"{i}\t{name}")
    print()

    print("EXPANDED_FIELDS")
    for i, name in enumerate(expanded, start=1):
        print(f"{i}\t{name}")


if __name__ == "__main__":
    main()
