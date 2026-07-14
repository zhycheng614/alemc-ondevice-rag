"""Fetch dataset rows via the HuggingFace datasets-server REST API.

This avoids the `datasets` + `pyarrow` dependency, which has no ARM64 Windows
wheel (it fails to build on the Snapdragon X Elite host). Plain urllib only.

The datasets-server paginates at <=100 rows/request; we loop with offset.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request

_BASE = "https://datasets-server.huggingface.co/rows"
_HEADERS = {"User-Agent": "Mozilla/5.0 (ALEMC-experiment)"}


def fetch_rows(dataset: str, config: str, split: str, n: int,
               page: int = 100, retries: int = 5, throttle: float = 0.4):
    """Yield up to `n` `row` dicts from a HF dataset via the REST API.

    Stops early (rather than raising) if the split is exhausted or the server
    returns no rows, so callers get as much as is available.
    """
    got = 0
    offset = 0
    while got < n:
        length = min(page, n - got)
        qs = urllib.parse.urlencode(dict(
            dataset=dataset, config=config, split=split,
            offset=offset, length=length))
        url = f"{_BASE}?{qs}"
        rows = _get_with_retry(url, retries)
        if not rows:
            break
        for r in rows:
            yield r["row"]
            got += 1
            if got >= n:
                return
        offset += len(rows)
        if len(rows) < length:
            break  # exhausted split
        time.sleep(throttle)  # be polite; avoid 429s


def _get_with_retry(url: str, retries: int):
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.load(resp)
            return data.get("rows", [])
        except urllib.error.HTTPError as e:  # noqa: PERF203
            last = e
            # 429 / 5xx: exponential backoff; 404: give up immediately.
            if e.code == 404:
                return []
            time.sleep(min(30.0, 3.0 * (2 ** attempt)))
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2.0 * (attempt + 1))
    # Exhausted retries: return empty so the caller uses what it has.
    print(f"[hf_rest] giving up on {url}: {last}", file=__import__("sys").stderr)
    return []
