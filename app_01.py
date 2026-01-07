from __future__ import annotations

import os
import json
import asyncio
from typing import List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from betting_bot import BettingBot

DATA_FILE = os.getenv("GR_DATA_FILE", "./gr_data_test.txt")
GCASES_FILE = os.getenv("GCASES_FILE", "./gcases_t.txt")
RCASES_FILE = os.getenv("RCASES_FILE", "./rcases_t.txt")
HISTORY_SIZE = 3000
RECALC_INTERVAL = 500

_recalc_task: Optional[asyncio.Task] = None


def count_occurrences_all(s: str, substring: str) -> int:
    """Count occurrences of substring in string using sliding window"""
    count = 0
    sub_len = len(substring)
    last = ""

    for i in range(len(s) - 2):
        last += s[i]
        if len(last) < sub_len:
            continue
        last = last[-sub_len:]

        if last == substring:
            count += 1

    return count


def generate_gr_strings(length: int = 5) -> List[str]:
    """Generate all possible strings of given length with 'g' and 'r'"""
    chars = ["g", "r"]
    result = []

    def backtrack(current: str):
        if len(current) == length:
            result.append(current)
            return
        for char in chars:
            backtrack(current + char)

    backtrack("")
    return result


def calculate_gcases(s: str) -> List[str]:
    """Calculate biased patterns (gcases) from history string"""
    print(
        f"Calculating gcases from {len(s)} chars, ~{round(len(s) / RECALC_INTERVAL)} cycles"
    )

    gcases = []
    rcases = []
    total_gwin = 0

    for i in range(8, 13):
        cases = generate_gr_strings(i)
        gwin = 0

        for c in cases:
            g = count_occurrences_all(s, f"{c}g")
            r = count_occurrences_all(s, f"{c}r")
            d = g - r

            p = -d if d > 0 else (r * 0.96 - g)

            # Filter conditions matching the JavaScript logic
            if d > 3 and (
                ((d / len(s)) * 100 > len(s) / 50000000)
                or ((d / len(s)) * 100 > len(s) / 100000000 and d * 3 > g)
            ):
                gwin += d
                gcases.append(c)

            if d < -3 and (
                ((p / len(s)) * 100 > len(s) / 50000000)
                or ((p / len(s)) * 100 > len(s) / 100000000 and p * 3 > g)
            ):
                gwin += d
                rcases.append(c)

        total_gwin += gwin

    print(f"Total gwin: {total_gwin}")

    # Sort by length descending
    gcases.sort(key=lambda x: len(x), reverse=True)
    rcases.sort(key=lambda x: len(x), reverse=True)
    return gcases, rcases


def predict_next_from_gcases(
    history: str, gcases: List[str], rcases: List[str]
) -> Tuple[str, str]:
    """Predict next character based on gcases patterns"""
    for pattern in gcases:
        if history.endswith(pattern):
            return "g", pattern
    for pattern in rcases:
        if history.endswith(pattern):
            return "r", pattern
    return "", ""


def read_history(path: str, n_chars: int) -> str:
    """Read the last n_chars from file"""
    if not os.path.exists(path):
        return ""

    with open(path, "rb") as f:
        # try:
        #     f.seek(0, os.SEEK_END)
        #     # size = f.tell()
        #     # f.seek(max(0, size - n_chars))
        # except OSError:
        #     f.seek(0)
        data = f.read()

    text = data.decode("utf-8", errors="ignore")
    text = "".join(ch for ch in text.lower() if ch in ("g", "r"))
    # return text[-n_chars:]
    return text


def save_history(path: str, history: str):
    """Save history to file, keeping only HISTORY_SIZE chars"""
    # history = history[-HISTORY_SIZE:]
    with open(path, "w", encoding="utf-8") as f:
        f.write(history)
    print("Save history to file")


def load_cases(path: str) -> List[str]:
    """Load gcases from file"""
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_cases(path: str, gcases: List[str]):
    """Save gcases to file"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gcases, f)


async def _recalc_gcases_in_thread(history_snapshot: str):
    """Run calculate_gcases in a worker thread, then commit results safely."""
    global _gcases, _rcases, _recalc_task

    try:
        new_gcases, new_rcases = await asyncio.to_thread(
            calculate_gcases, history_snapshot
        )

        async with _lock:
            _gcases = new_gcases
            _rcases = new_rcases
            save_cases(GCASES_FILE, _gcases)
            save_cases(RCASES_FILE, _rcases)
            # Save latest history (not the snapshot) so you persist everything received
            save_history(DATA_FILE, _history)

    finally:
        # allow future recalcs to be scheduled
        _recalc_task = None


# Global state
_lock = asyncio.Lock()
_history = ""
_gcases = []
_rcases = []
_predict_count = 0

_bot = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic"""
    global _history, _gcases, _rcases, _bot

    # Startup: read history and gcases
    _history = read_history(DATA_FILE, HISTORY_SIZE)
    _gcases = load_cases(GCASES_FILE)
    _rcases = load_cases(RCASES_FILE)

    # If no gcases or history changed significantly, recalculate
    if not _gcases and len(_history) > 1000:
        print("Calculating initial gcases...")
        _gcases, _rcases = calculate_gcases(_history)
        save_cases(GCASES_FILE, _gcases)
        save_cases(RCASES_FILE, _rcases)

    print(f"Loaded {len(_history)} chars of history")
    print(f"Loaded {len(_gcases)} gcases patterns")
    print(f"Loaded {len(_rcases)} rcases patterns")

    history = list(_history[-40:])
    _bot = BettingBot(history, gcases=_gcases, rcases=_rcases, max_bet=32)

    yield

    # Shutdown: save history
    save_history(DATA_FILE, _history)


app = FastAPI(title="Trenball Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/next/{cur}")
async def predict(cur: str):
    """Predict next character and update history"""
    global _history, _gcases, _rcases, _predict_count, _recalc_task, _bot

    cur = cur.lower().strip()
    if cur not in ("g", "r"):
        raise HTTPException(status_code=400, detail='current must be "g" or "r"')

    async with _lock:
        # Update history
        _history += cur
        # _history = _history[-HISTORY_SIZE:]

        # Increment predict count
        _predict_count += 1

        # If interval hit: return empty next, start recalculation in background thread
        if _predict_count % RECALC_INTERVAL == 0:
            # snapshot history for this recalculation run
            history_snapshot = _history

            # prevent stacking multiple recalcs if requests pile up
            if _recalc_task is None or _recalc_task.done():
                _recalc_task = asyncio.create_task(
                    _recalc_gcases_in_thread(history_snapshot)
                )

            return {
                "next": "",
                "amount": 0,
                # "matched_pattern": "",
                "predict_count": _predict_count,
                "next_recalc_in": RECALC_INTERVAL,
                "recalc_started": True,
            }

        # Normal path: Predict next
        result = _bot.process_round(cur)
        # nxt, matched_pattern = predict_next_from_gcases(_history, _gcases, _rcases)
        nxt = result["predicted_next"]
        amount = result["next_bet"]

        return {
            "next": nxt,
            "amount": amount / 4,
            # "matched_pattern": matched_pattern,
            "predict_count": _predict_count,
            "next_recalc_in": RECALC_INTERVAL - (_predict_count % RECALC_INTERVAL),
            "recalc_started": False,
        }


@app.get("/api/stats")
async def get_stats():
    """Get current stats"""
    async with _lock:
        return {
            "history_length": len(_history),
            "gcases_count": len(_gcases),
            "rcases_count": len(_rcases),
            "predict_count": _predict_count,
            "next_recalc_in": RECALC_INTERVAL - (_predict_count % RECALC_INTERVAL),
        }


@app.get("/api/gcases")
async def get_gcases():
    """Get current gcases patterns"""
    async with _lock:
        return {"gcases": _gcases}


@app.get("/api/rcases")
async def get_rcases():
    """Get current rcases patterns"""
    async with _lock:
        return {"rcases": _rcases}


@app.post("/api/recalc")
async def force_recalc():
    """Force recalculation of gcases"""
    global _gcases, _rcases, _recalc_task

    async with _lock:
        print("\n=== Force recalculating gcases ===")
        if _recalc_task is None or _recalc_task.done():
            _recalc_task = asyncio.create_task(_recalc_gcases_in_thread(_history))

        return {"message": "Gcases recalculating"}


@app.get("/api")
def root():
    return {"message": "Trenball Predictor API is running 🚀"}


if __name__ == "__main__":
    uvicorn.run("app:app", port=10081, reload=False)
