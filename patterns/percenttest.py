from __future__ import annotations

import json
import os
import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Tuple

# DATA_FILE = os.getenv("GR_DATA_FILE", "./gr_data.txt")
# DATA_FILE = os.getenv("GR_DATA_FILE", "./gr_data_test.txt")
# DATA_FILE = os.getenv("GR_DATA_FILE", "./gr_trenball.txt")
DATA_FILE = os.getenv("GR_DATA_FILE", "./gr_trenball_test.txt")

HISTORY_SAVE = 100000
HISTORY_KEEP = 5000
WINDOW_N = 3000

MAX_LEN = 10
MIN_SKEW = 0.3
TOP_N = 200

_lock = asyncio.Lock()


def find_biased_patterns(
    s: str, max_len: int = 10, min_skew: float = 0.2, top_n: int = 200
):
    s = (s or "").strip().lower()
    if len(s) < 2:
        return []

    counts = defaultdict(lambda: [0, 0])

    for i in range(len(s) - 1):
        nxt = s[i + 1]
        if nxt not in ("g", "r"):
            continue

        nxt_idx = 0 if nxt == "g" else 1

        start = max(0, i - max_len + 1)
        for j in range(i, start - 1, -1):
            ctx = s[j : i + 1]
            if ("g" in ctx) and ("r" in ctx):
                counts[ctx][nxt_idx] += 1

    results = []
    for ctx, (g, r) in counts.items():
        n = g + r
        if n <= 10:
            continue

        skew = abs(g - r) / n
        if skew < min_skew:
            continue

        results.append({"pattern": ctx, "support": n, "g": g, "r": r, "skew": skew})

    results.sort(key=lambda d: (d["skew"], d["support"]), reverse=True)
    return results[:top_n]


def predict_next_from_patterns(
    history: str, patterns: List[Dict[str, Any]]
) -> Tuple[str, str]:
    for row in patterns:
        pat = row["pattern"]
        if history.endswith(pat):
            if row["g"] > row["r"]:
                print("g", row)
                return "g", row
            if row["r"] > row["g"]:
                print("r", row)
                return "r", row
            return "", row
    return "", ""


def read_tail(path: str, n_chars: int, save_chars: int) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r+b") as f:
        try:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size > save_chars:
                f.seek(save_chars)
                f.truncate()
                size = save_chars
            f.seek(max(0, size - n_chars))
        except OSError:
            f.seek(0)
        data = f.read()

    text = data.decode("utf-8", errors="ignore")
    text = "".join(ch for ch in text.lower() if ch in ("g", "r"))
    return text[-n_chars:]


async def predict(cur):
    global _history

    print(cur)

    if cur not in ("g", "r"):
        return False

    async with _lock:
        with open(DATA_FILE, "a", encoding="utf-8") as f:
            f.write(cur)

        if len(_history) > HISTORY_KEEP:
            _history = _history[-HISTORY_KEEP:]

        window = _history[-WINDOW_N:]

        patterns = find_biased_patterns(
            window,
            max_len=MAX_LEN,
            min_skew=MIN_SKEW,
            top_n=TOP_N,
        )
        _history += cur
        window += cur

        nxt, matched = predict_next_from_patterns(window, patterns)

    return nxt
    # {"next": nxt, "matched_pattern": matched}


_history = ""

if __name__ == "__main__":
    _history = read_tail(DATA_FILE, HISTORY_KEEP, HISTORY_SAVE)

    #   start save json to str
    data = []
    filename="2025-01-04T00_2026-01-10T23_raw"
    with open(f"tren/{filename}.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    gr_data = []

    for item in data:
        color = "r" if item["color"] == "red" else "g"
        gr_data.append(color)
    text = "".join(gr_data)
    with open(f"gr_tren/{filename}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    #   end save json to str

    # gr_data = ""

    # with open("gr_limbo.txt", "r", encoding="utf-8", errors="ignore") as f:
    #     gr_data = f.read()

    # total = len(gr_data)
    # bet = 0
    # gwin = 0
    # rwin = 0

    # for i in range(len(gr_data)):
    #     if i == 0:
    #         continue
    #     cur = gr_data[i - 1]
    #     next_cur = gr_data[i]
    #     next_predict = asyncio.run(predict(cur))

    #     if next_predict != "":
    #         bet += 1
    #         if next_predict == next_cur:
    #             if next_predict == "g":
    #                 gwin += 1
    #             else:
    #                 rwin += 1

    # print(f"Total: {total}, Bet: {bet}, Gwin: {gwin}, Rwin: {rwin}")
