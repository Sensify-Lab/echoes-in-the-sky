from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_bluesky")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg_cache_bluesky")

import matplotlib

matplotlib.use("Agg")

import duckdb
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns


BASE_DIR = Path("local_data/03_Github_data/08_EO_effect_structure_changes")
TABLE_DIR = BASE_DIR / "tables"
FIG_DIR = BASE_DIR / "publication_figures"
PER_THEME_DIR = FIG_DIR / "per_theme"

PARQUET_GLOB = "local_data/03_Github_data/04_final_merged_10M/*.parquet"
MAX_BREAKS = 6
MIN_SEGMENT_SHARE = 0.15
CHI2_95 = 3.841458820694124


def ensure_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PER_THEME_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    return "".join(c.lower() if c.isalnum() else "_" for c in text).strip("_")


def build_theme_daily() -> pd.DataFrame:
    con = duckdb.connect()
    query = f"""
        WITH base AS (
            SELECT
                CAST(try_cast("record.created_at" AS TIMESTAMPTZ) AS DATE) AS day,
                Theme_Name AS bluesky_theme
            FROM read_parquet('{PARQUET_GLOB}')
            WHERE try_cast("record.created_at" AS TIMESTAMPTZ) IS NOT NULL
              AND Theme_Name IS NOT NULL
        )
        SELECT
            day,
            bluesky_theme,
            count(*) AS posts
        FROM base
        GROUP BY 1, 2
        ORDER BY 2, 1
    """
    df = con.execute(query).df()
    con.close()
    df["day"] = pd.to_datetime(df["day"])
    df.to_csv(TABLE_DIR / "theme_daily_posts_sparse.csv", index=False)

    full_rows = []
    for theme, sub in df.groupby("bluesky_theme", sort=True):
        idx = pd.date_range(sub["day"].min(), sub["day"].max(), freq="D")
        full = (
            sub.set_index("day")["posts"]
            .reindex(idx, fill_value=0)
            .rename_axis("day")
            .reset_index()
        )
        full["bluesky_theme"] = theme
        full_rows.append(full)
    out = pd.concat(full_rows, ignore_index=True)
    out = out.rename(columns={0: "posts"})
    out["posts"] = out["posts"].astype(int)
    out = out[["day", "bluesky_theme", "posts"]]
    out.to_csv(TABLE_DIR / "theme_daily_posts_full.csv", index=False)
    return out


def segment_rss(y: np.ndarray, start: int, end: int) -> float:
    seg = y[start:end]
    if len(seg) == 0:
        return 0.0
    mean = float(seg.mean())
    return float(((seg - mean) ** 2).sum())


def total_rss(y: np.ndarray, break_ends: list[int]) -> float:
    prev = 0
    rss = 0.0
    for bp in break_ends:
        rss += segment_rss(y, prev, bp)
        prev = bp
    return rss


def choose_breaks(y: np.ndarray, min_size: int) -> tuple[list[int], float]:
    n = len(y)
    max_allowed = min(MAX_BREAKS, max(0, n // min_size - 1))
    algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(y)

    candidates: list[tuple[int, list[int], float, float]] = []
    rss0 = total_rss(y, [n])
    bic0 = n * math.log(max(rss0 / n, 1e-9)) + 1 * math.log(n)
    candidates.append((0, [n], rss0, bic0))

    for k in range(1, max_allowed + 1):
        try:
            bkps = algo.predict(n_bkps=k)
        except Exception:
            continue
        rss = total_rss(y, bkps)
        bic = n * math.log(max(rss / n, 1e-9)) + (k + 1) * math.log(n)
        candidates.append((k, bkps, rss, bic))

    best = min(candidates, key=lambda x: x[3])
    return best[1], best[2]


def confidence_interval_indices(y: np.ndarray, break_ends: list[int], j: int, min_size: int, rss_best: float) -> tuple[int, int]:
    n = len(y)
    segments = len(break_ends)
    sigma2 = max(rss_best / max(n - segments, 1), 1e-9)
    threshold = rss_best + CHI2_95 * sigma2

    bp = break_ends[j]
    prev_end = 0 if j == 0 else break_ends[j - 1]
    next_end = n if j == len(break_ends) - 1 else break_ends[j + 1]

    low = prev_end + min_size
    high = next_end - min_size
    accepted: list[int] = []

    for cand in range(low, high + 1):
        trial = break_ends.copy()
        trial[j] = cand
        trial = sorted(trial)
        if any((trial[i] - (0 if i == 0 else trial[i - 1])) < min_size for i in range(len(trial))):
            continue
        if (n - trial[-1]) < min_size:
            continue
        rss = total_rss(y, trial)
        if rss <= threshold:
            accepted.append(cand)

    if not accepted:
        return bp, bp
    return min(accepted), max(accepted)


def analyze_theme(theme: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = sub.sort_values("day").reset_index(drop=True)
    y = sub["posts"].to_numpy(dtype=float)
    n = len(y)
    min_size = max(2, int(math.ceil(n * MIN_SEGMENT_SHARE)))
    break_ends, rss_best = choose_breaks(y, min_size)

    summary = {
        "bluesky_theme": theme,
        "n_days": n,
        "min_segment_days": min_size,
        "n_breaks": max(len(break_ends) - 1, 0),
        "series_start": sub["day"].iloc[0],
        "series_end": sub["day"].iloc[-1],
    }

    rows = []
    for j, bp in enumerate(break_ends[:-1]):
        ci_low_idx, ci_high_idx = confidence_interval_indices(y, break_ends, j, min_size, rss_best)
        rows.append(
            {
                "bluesky_theme": theme,
                "Breakpoint": j + 1,
                "CI_2.5": ci_low_idx + 1,
                "Estimate": bp,
                "CI_97.5": ci_high_idx + 1,
                "CI_2.5_date": sub["day"].iloc[ci_low_idx],
                "Break_date": sub["day"].iloc[bp - 1],
                "CI_97.5_date": sub["day"].iloc[ci_high_idx],
                "segment_min_days": min_size,
            }
        )

    return pd.DataFrame([summary]), pd.DataFrame(rows)


def plot_theme(theme: str, sub: pd.DataFrame, breaks_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(sub["day"], sub["posts"], color="#1f77b4", linewidth=1.8, label="Daily posts")
    ax.set_title(f"{theme}: Daily Posts with Structural Breaks", loc="left", fontsize=18)
    ax.set_ylabel("Posts per day")
    ax.set_xlabel("Date")
    ax.grid(True, axis="y", alpha=0.25)
    for _, row in breaks_df.iterrows():
        ax.axvspan(pd.to_datetime(row["CI_2.5_date"]), pd.to_datetime(row["CI_97.5_date"]), color="#d62728", alpha=0.12)
        ax.axvline(pd.to_datetime(row["Break_date"]), color="#7f0000", linewidth=2.2, linestyle="--")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()

    stem = slugify(theme)
    fig.savefig(PER_THEME_DIR / f"{stem}_breaks.png", dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(PER_THEME_DIR / f"{stem}_breaks.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    theme_daily = build_theme_daily()

    summary_parts = []
    break_parts = []
    for theme, sub in theme_daily.groupby("bluesky_theme", sort=True):
        theme_summary, theme_breaks = analyze_theme(theme, sub)
        summary_parts.append(theme_summary)
        break_parts.append(theme_breaks)
        plot_theme(theme, sub, theme_breaks)

    summary_df = pd.concat(summary_parts, ignore_index=True).sort_values("bluesky_theme")
    breaks_df = pd.concat(break_parts, ignore_index=True).sort_values(["bluesky_theme", "Breakpoint"])

    summary_df.to_csv(TABLE_DIR / "theme_structure_break_summary.csv", index=False)
    breaks_df.to_csv(TABLE_DIR / "theme_structure_breaks.csv", index=False)

    print(f"daily_full\t{TABLE_DIR / 'theme_daily_posts_full.csv'}")
    print(f"summary\t{TABLE_DIR / 'theme_structure_break_summary.csv'}")
    print(f"breaks\t{TABLE_DIR / 'theme_structure_breaks.csv'}")
    print(f"figures\t{PER_THEME_DIR}")


if __name__ == "__main__":
    main()
