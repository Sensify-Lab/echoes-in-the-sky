from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns


SOURCE_SCRIPT = Path(__file__).with_name("structure_break_fixed_ci.py")
OUTPUT_BASE = Path(
    "local_data/03_Github_data/08_EO_effect_structure_changes/"
    "post_2025_01_20_max20_bootstrap_ci_two_colors"
)
BOOTSTRAP_REPS = 120
BOOTSTRAP_SEED = 42


def load_base_module():
    spec = importlib.util.spec_from_file_location("structure_break_base", SOURCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


base = load_base_module()
base.BASE_DIR = OUTPUT_BASE
base.TABLE_DIR = OUTPUT_BASE / "tables"
base.FIG_DIR = OUTPUT_BASE / "publication_figures"
base.PER_THEME_DIR = base.FIG_DIR / "per_theme"


def fitted_piecewise_means(y: np.ndarray, break_ends: list[int]) -> np.ndarray:
    fitted = np.zeros_like(y, dtype=float)
    prev = 0
    for bp in break_ends:
        seg = y[prev:bp]
        seg_mean = float(seg.mean()) if len(seg) else 0.0
        fitted[prev:bp] = max(seg_mean, 1e-6)
        prev = bp
    return fitted


def bootstrap_break_intervals(
    y: np.ndarray,
    break_ends: list[int],
    min_size: int,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
) -> list[tuple[int, int]]:
    n_breaks = max(len(break_ends) - 1, 0)
    if n_breaks == 0:
        return []

    rng = np.random.default_rng(seed)
    fitted = fitted_piecewise_means(y, break_ends)
    collected: list[list[int]] = [[] for _ in range(n_breaks)]

    for _ in range(reps):
        y_boot = rng.poisson(fitted).astype(float)
        try:
            bkps = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(y_boot).predict(n_bkps=n_breaks)
        except Exception:
            continue
        if len(bkps) != n_breaks + 1:
            continue
        for j, bp in enumerate(bkps[:-1]):
            collected[j].append(int(bp))

    out: list[tuple[int, int]] = []
    for j, samples in enumerate(collected):
        if len(samples) < 20:
            bp = int(break_ends[j])
            out.append((bp, bp))
            continue
        low = int(round(np.percentile(samples, 2.5)))
        high = int(round(np.percentile(samples, 97.5)))
        low = max(low, 1)
        high = min(high, len(y) - 1)
        if low > high:
            low, high = high, low
        out.append((low, high))
    return out


def analyze_theme(theme: str, sub: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = sub.sort_values("day").reset_index(drop=True)
    y = sub["posts"].to_numpy(dtype=float)
    n = len(y)
    min_size = max(2, int(np.ceil(n * base.MIN_SEGMENT_SHARE)))
    break_ends, _ = base.choose_breaks(y, min_size)
    ci_bounds = bootstrap_break_intervals(y, break_ends, min_size)

    summary = {
        "bluesky_theme": theme,
        "n_days": n,
        "min_segment_days": min_size,
        "n_breaks": max(len(break_ends) - 1, 0),
        "series_start": sub["day"].iloc[0],
        "series_end": sub["day"].iloc[-1],
        "ci_method": "bootstrap_poisson_piecewise_mean_95pct",
        "bootstrap_reps": BOOTSTRAP_REPS,
    }

    rows = []
    for j, bp in enumerate(break_ends[:-1]):
        ci_low_idx, ci_high_idx = ci_bounds[j]
        rows.append(
            {
                "bluesky_theme": theme,
                "Breakpoint": j + 1,
                "CI_2.5": ci_low_idx,
                "Estimate": bp,
                "CI_97.5": ci_high_idx,
                "CI_2.5_date": base.breakpoint_date_from_end_index(sub, ci_low_idx),
                "Break_date": base.breakpoint_date_from_end_index(sub, bp),
                "CI_97.5_date": base.breakpoint_date_from_end_index(sub, ci_high_idx),
                "segment_min_days": min_size,
                "ci_method": "bootstrap_poisson_piecewise_mean_95pct",
                "bootstrap_reps": BOOTSTRAP_REPS,
            }
        )

    return pd.DataFrame([summary]), pd.DataFrame(rows)


def plot_theme(theme: str, sub: pd.DataFrame, breaks_df: pd.DataFrame) -> None:
    sns.set_theme(style="white", context="talk")
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

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(sub["day"], sub["posts"], color="#1f77b4", linewidth=1.8, label="Daily posts", zorder=3)
    ax.set_title(f"{theme}: Daily Posts After January 20, 2025", loc="left", fontsize=18)
    ax.set_ylabel("Posts per day")
    ax.set_xlabel("Month-Year")
    ax.grid(False)

    segment_bounds = [sub["day"].iloc[0]]
    segment_bounds.extend(pd.to_datetime(breaks_df["Break_date"]).tolist())
    segment_bounds.append(sub["day"].iloc[-1])
    segment_colors = ["#fff1cc", "#dff3e3"]
    for i in range(len(segment_bounds) - 1):
        ax.axvspan(
            segment_bounds[i],
            segment_bounds[i + 1],
            color=segment_colors[i % 2],
            alpha=0.42,
            zorder=0,
        )

    ymax = max(float(sub["posts"].max()), 1.0)
    label_levels = [0.97, 0.90, 0.83]
    for idx, (_, row) in enumerate(breaks_df.iterrows()):
        ci_low = pd.to_datetime(row["CI_2.5_date"])
        brk = pd.to_datetime(row["Break_date"])
        ci_high = pd.to_datetime(row["CI_97.5_date"])

        ci_left = ci_low
        ci_right = ci_high
        if ci_left == ci_right:
            ci_left = ci_left - pd.Timedelta(days=0.75)
            ci_right = ci_right + pd.Timedelta(days=0.75)

        label_y = ymax * label_levels[idx % len(label_levels)]
        label_y_mid = max(label_y - ymax * 0.06, ymax * 0.68)

        ax.axvspan(ci_left, ci_right, color="#c0392b", alpha=0.12, zorder=1)
        ax.axvline(ci_low, color="#8e2a1e", linewidth=1.8, linestyle="-", zorder=4)
        ax.axvline(brk, color="#6e0000", linewidth=2.5, linestyle="--", zorder=5)
        ax.axvline(ci_high, color="#8e2a1e", linewidth=1.8, linestyle="-", zorder=4)

        ax.text(
            ci_low,
            label_y,
            f"CI low\n{ci_low.strftime('%Y-%m-%d')}",
            color="#6e0000",
            fontsize=8,
            ha="right",
            va="top",
        )
        ax.text(
            brk,
            label_y_mid,
            f"Break\n{brk.strftime('%Y-%m-%d')}",
            color="#6e0000",
            fontsize=8,
            ha="center",
            va="top",
        )
        ax.text(
            ci_high,
            label_y,
            f"CI high\n{ci_high.strftime('%Y-%m-%d')}",
            color="#6e0000",
            fontsize=8,
            ha="left",
            va="top",
        )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()

    stem = base.slugify(theme)
    fig.savefig(
        base.PER_THEME_DIR / f"{stem}_breaks_post_2025_01_20_max20_bootstrap_ci_two_colors.png",
        dpi=320,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        base.PER_THEME_DIR / f"{stem}_breaks_post_2025_01_20_max20_bootstrap_ci_two_colors.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


base.analyze_theme = analyze_theme
base.plot_theme = plot_theme


if __name__ == "__main__":
    base.main()
