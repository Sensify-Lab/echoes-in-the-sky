from __future__ import annotations

import shutil
from pathlib import Path

BASE_DIR = Path("local_data/03_Github_data/05_bluesky_data_merged_sentiment")
SRC_FIG_DIR = BASE_DIR / "text_sentiment_analysis_outputs" / "publication_figures"
DEST_DIR = BASE_DIR / "final_text_sentiment_figures"

SELECTED = [
    "text_sentiment_distribution",
    "monthly_text_sentiment_lines",
    "monthly_text_sentiment_share_area",
    "theme_by_text_sentiment_heatmap",
    "theme_sentiment_share_heatmap",
    "engagement_by_text_sentiment",
    "text_vs_media_sentiment_heatmap",
    "theme_mismatch_rate_barh",
]

def main() -> None:
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    manifest_lines: list[str] = []

    for stem in SELECTED:
        for ext in ("png", "pdf"):
            src = SRC_FIG_DIR / f"{stem}.{ext}"
            dst = DEST_DIR / f"{stem}.{ext}"
            if src.exists():
                shutil.copy2(src, dst)
                manifest_lines.append(dst.name)

    (DEST_DIR / "manifest.txt").write_text("\n".join(sorted(manifest_lines)) + "\n", encoding="utf-8")

    print(f"saved_final_package\t{DEST_DIR}")
    print(f"selected_figures\t{len(SELECTED)}")

if __name__ == "__main__":
    main()
