"""
Adaptive time-slicing Bluesky search collector with retries.

This script is still workspace-oriented rather than library-oriented. Update the
configuration block below before running it in a new environment.
"""

import json
import gzip
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from datetime import datetime, timezone, timedelta

import pandas as pd
from atproto import Client


# --------------------
# Config
# --------------------
QUERIES = [
    "trump",
    "maga",
    "trumpism",
    "potus45",
    "america first",
    "keep america great",
    "Trumpocalypse",
    "Republican",
    "Drumpf",
    "45th President",
]

LIMIT = 100
SORT = "latest"
SLEEP_SECONDS = 0.2

START_DATE_STR = "01-01-2025"   # MM-DD-YYYY
END_DATE_STR = "01-02-2025"

# START_DT = "2024-09-11T00:00:00Z"
# END_DT ="2024-09-11T00:00:00Z"

INITIAL_WINDOW_DAYS = 0.2
MIN_WINDOW_HOURS = 0.001
SHRINK_FACTOR = 0.5
MAX_PAGES_PER_QUERY_PER_WINDOW = 100

# Retry behavior for API failures
MAX_RETRIES = 999
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2
BACKOFF_MAX_SECONDS = 3600.0

START_DT = datetime.strptime(START_DATE_STR, "%m-%d-%Y").replace(
    hour=0, minute=0, second=0, tzinfo=timezone.utc
)
END_DT = datetime.strptime(END_DATE_STR, "%m-%d-%Y").replace(
    hour=23, minute=59, second=59, tzinfo=timezone.utc
)

OUT_DIR = Path("00_Raw_data_2026/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_JSONL = OUT_DIR / f"test_bluesky_posts_{START_DATE_STR}_to_{END_DATE_STR}_full.jsonl.gz"
CORE_CSV = OUT_DIR / f"test_bluesky_posts_{START_DATE_STR}_to_{END_DATE_STR}_core.csv"

BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD", "")


# --------------------
# Helpers
# --------------------
def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def to_plain(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if hasattr(obj, "__dict__"):
        return {k: to_plain(v) for k, v in vars(obj).items() if not k.startswith("_")}
    return str(obj)


def append_jsonl(posts: Iterable[Any], out_file: Path) -> int:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with gzip.open(out_file, "at", encoding="utf-8") as f:
        for p in posts:
            f.write(json.dumps(to_plain(p), ensure_ascii=False) + "\n")
            n += 1
    return n


def append_core_csv(rows: List[Dict[str, Any]], out_file: Path) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not out_file.exists()
    df.to_csv(out_file, mode="a", index=False, header=write_header)


def posts_to_core_rows(posts: Iterable[Any], uri_to_queries: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for post in posts:
        r = getattr(post, "record", None)
        author = getattr(post, "author", None)
        uri = getattr(post, "uri", None)

        row: Dict[str, Any] = {
            "uri": uri,
            "cid": getattr(post, "cid", None),
            "indexed_at": getattr(post, "indexed_at", None),
            "created_at": getattr(r, "created_at", None) if r else None,

            "author_handle": getattr(author, "handle", None),
            "author_display_name": getattr(author, "display_name", None),
            "author_did": getattr(author, "did", None),
            "author_created_at": getattr(author, "created_at", None),

            "text": getattr(r, "text", None) if r else None,
            "langs": getattr(r, "langs", None) if r else None,

            "reply_parent_uri": getattr(getattr(getattr(r, "reply", None), "parent", None), "uri", None) if r else None,
            "reply_root_uri": getattr(getattr(getattr(r, "reply", None), "root", None), "uri", None) if r else None,

            "post_moderation_labels": getattr(r, "labels", None) if r else None,

            "like_count": getattr(post, "like_count", None),
            "reply_count": getattr(post, "reply_count", None),
            "repost_count": getattr(post, "repost_count", None),
            "quote_count": getattr(post, "quote_count", None),
            "bookmark_count": getattr(post, "bookmark_count", None),

            "matched_queries": "|".join(sorted(uri_to_queries.get(uri, set()))) if uri else "",
            "matched_query_count": len(uri_to_queries.get(uri, set())) if uri else 0,
        }

        if r and getattr(r, "embed", None) and hasattr(r.embed, "external"):
            row["embed_media_title"] = getattr(r.embed.external, "title", None)
            row["embed_media_url"] = getattr(r.embed.external, "uri", None)
            row["embed_media_description"] = getattr(r.embed.external, "description", None)
        else:
            row["embed_media_title"] = None
            row["embed_media_url"] = None
            row["embed_media_description"] = None

        rows.append(row)
    return rows


def retry_search_posts(
    client: Client,
    params: Dict[str, Any],
    *,
    max_retries: int = MAX_RETRIES,
) -> Any:
    """
    Retry wrapper around client.app.bsky.feed.search_posts.
    Raises the last exception if all retries fail.
    """
    attempt = 0
    while True:
        try:
            return client.app.bsky.feed.search_posts(params=params)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise

            sleep_s = min(
                BACKOFF_BASE_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1)),
                BACKOFF_MAX_SECONDS,
            )
            print(f"[WARN] search_posts failed (attempt {attempt}/{max_retries}): {type(e).__name__}: {e}")
            print(f"[WARN] Sleeping {sleep_s:.1f}s then retrying...")
            time.sleep(sleep_s)


def fetch_window_for_query_staged(
    client: Client,
    query: str,
    window_start: datetime,
    window_end: datetime,
    seen_global: Set[str],
    seen_staged: Set[str],
    uri_to_queries_staged: Dict[str, Set[str]],
) -> Tuple[List[Any], int, bool]:
    """
    Returns: (new_posts, pages_used, overflow_or_error)
    overflow_or_error True means redo this window.
    """
    cursor: Optional[str] = None
    pages = 0
    new_posts: List[Any] = []

    while True:
        pages += 1

        params = {
            "q": query,
            "limit": LIMIT,
            "sort": SORT,
            "since": iso(window_start),
            "until": iso(window_end),
            "cursor": cursor,
        }

        try:
            data = retry_search_posts(client, params)
        except Exception as e:
            print(f"[ERROR] search_posts ultimately failed for query='{query}' window={window_start}..{window_end}")
            print(f"[ERROR] {type(e).__name__}: {e}")
            return [], pages, True

        for post in getattr(data, "posts", []):
            uri = getattr(post, "uri", None)
            if not uri:
                continue

            uri_to_queries_staged.setdefault(uri, set()).add(query)

            if uri in seen_global or uri in seen_staged:
                continue

            seen_staged.add(uri)
            new_posts.append(post)

        cursor = getattr(data, "cursor", None)

        if pages >= MAX_PAGES_PER_QUERY_PER_WINDOW:
            return new_posts, pages, True

        if not cursor:
            return new_posts, pages, False

        time.sleep(SLEEP_SECONDS)


def collect_window_once(
    client: Client,
    window_start: datetime,
    window_end: datetime,
    seen_uris: Set[str],
    uri_to_queries: Dict[str, Set[str]],
) -> Tuple[int, bool]:
    """
    Returns: (new_saved_rows, redo_window)
    redo_window True if overflow or error happened.
    """
    seen_staged: Set[str] = set()
    uri_to_queries_staged: Dict[str, Set[str]] = {}
    posts_staged: List[Any] = []

    redo_window = False

    for q in QUERIES:
        new_posts, pages_used, overflow_or_error = fetch_window_for_query_staged(
            client,
            q,
            window_start,
            window_end,
            seen_uris,
            seen_staged,
            uri_to_queries_staged,
        )

        print(
            f"Window {window_start} to {window_end} | "
            f"query '{q}' pages {pages_used} | new {len(new_posts)} | redo {overflow_or_error}"
        )

        posts_staged.extend(new_posts)

        if overflow_or_error:
            redo_window = True
            break

        time.sleep(SLEEP_SECONDS)

    if redo_window:
        return 0, True

    seen_uris.update(seen_staged)
    for uri, qs in uri_to_queries_staged.items():
        uri_to_queries.setdefault(uri, set()).update(qs)

    if posts_staged:
        append_jsonl(posts_staged, RAW_JSONL)
        core_rows = posts_to_core_rows(posts_staged, uri_to_queries)
        append_core_csv(core_rows, CORE_CSV)
        return len(core_rows), False

    return 0, False


def check_created_at(infile_path: Path) -> None:
    df = pd.read_json(infile_path, lines=True, compression="gzip")

    if "record.created_at" in df.columns:
        created = df["record.created_at"]
    elif "record" in df.columns:
        created = df["record"].apply(lambda r: r.get("created_at") if isinstance(r, dict) else None)
    else:
        raise ValueError("No created_at found in record")

    created = pd.to_datetime(created, errors="coerce", utc=True).dropna()

    if created.empty:
        print("No valid created_at values found.")
        return

    print("Earliest post:", created.min())
    print("Latest post:", created.max())


# --------------------
# Main adaptive slicing
# --------------------
def main() -> None:
    if not BLUESKY_HANDLE or not BLUESKY_APP_PASSWORD:
        raise RuntimeError(
            "Missing Bluesky credentials. Set BLUESKY_HANDLE and "
            "BLUESKY_APP_PASSWORD in the environment before running this script."
        )

    client = Client()
    client.login(BLUESKY_HANDLE, BLUESKY_APP_PASSWORD)

    seen_uris: Set[str] = set()
    uri_to_queries: Dict[str, Set[str]] = {}

    current_start = START_DT
    base_window = timedelta(days=INITIAL_WINDOW_DAYS)
    min_window = timedelta(hours=MIN_WINDOW_HOURS)

    while current_start < END_DT:
        window = min(base_window, END_DT - current_start)

        while True:
            current_end = current_start + window
            if current_end > END_DT:
                current_end = END_DT

            print(f"\nCollecting window: {current_start} to {current_end} | duration {current_end - current_start}")

            total_new, redo_window = collect_window_once(
                client, current_start, current_end, seen_uris, uri_to_queries
            )

            print(f"Window done | new saved {total_new} | unique total {len(seen_uris)}")

            if not redo_window:
                current_start = current_end
                break

            if window <= min_window:
                print("Reached MIN_WINDOW_HOURS and still needs redo.")
                print("Moving forward to avoid infinite loop.")
                current_start = current_end
                break

            new_seconds = int(window.total_seconds() * SHRINK_FACTOR)
            window = max(timedelta(seconds=new_seconds), min_window)

            print(f"Redo requested. Retry same start with smaller window: {window}")
            time.sleep(SLEEP_SECONDS)

    print("\nDone.")
    print("Raw JSONL:", RAW_JSONL)
    print("Core CSV:", CORE_CSV)
    print("Total unique posts:", len(seen_uris))


if __name__ == "__main__":
    t0 = time.time()
    main()
    check_created_at(RAW_JSONL)
    t1 = time.time()
    print("Time Spent:", (t1 - t0) / 60, "minute")
