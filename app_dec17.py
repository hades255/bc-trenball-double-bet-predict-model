from __future__ import annotations

import asyncio
from collections import Counter
from typing import Dict, Tuple
from dataclasses import dataclass
import pickle
from contextlib import asynccontextmanager

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

_lock = asyncio.Lock()
_history = ""
model = None


def _clean_gr(s: str) -> str:
    """Keep only g/r (lowercased)."""
    return "".join(ch for ch in (s or "").lower() if ch in ("g", "r"))


@dataclass
class NGramGRModel:
    order: int  # context length
    # counts[context] -> Counter({'g': n, 'r': n})
    counts: Dict[str, Counter]
    # overall -> Counter({'g': n, 'r': n})
    overall: Counter

    def predict_next(self, history: str) -> Tuple[str, float, Dict[str, float]]:
        """
        Predict next char from history.
        Returns: (prediction, confidence, probs_dict)
        """
        h = _clean_gr(history)

        # Try longest context -> shorter -> unigram
        for k in range(min(self.order, len(h)), -1, -1):
            ctx = h[-k:] if k > 0 else ""  # empty context = global stats
            c = self.counts.get(ctx)
            if c and (c["g"] + c["r"] > 0):
                total = c["g"] + c["r"]
                pg = c["g"] / total
                pr = c["r"] / total

                # deterministic tie-break using overall frequencies
                if pg > pr:
                    return "g", pg, {"g": pg, "r": pr}
                if pr > pg:
                    return "r", pr, {"g": pg, "r": pr}

                # tie
                og, or_ = self.overall["g"], self.overall["r"]
                pred = "g" if og >= or_ else "r"
                conf = pg  # same as pr
                return pred, conf, {"g": pg, "r": pr}

        # absolute fallback (shouldn’t happen)
        total = self.overall["g"] + self.overall["r"]
        if total == 0:
            return "g", 0.5, {"g": 0.5, "r": 0.5}
        pg = self.overall["g"] / total
        pr = self.overall["r"] / total
        pred = "g" if pg >= pr else "r"
        conf = max(pg, pr)
        return pred, conf, {"g": pg, "r": pr}


def load_model(path: str = "gr_predictor.pkl") -> NGramGRModel:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    # If old pkl stored the dataclass directly, try to convert it
    if isinstance(obj, NGramGRModel):
        return obj

    # If pkl stored a dict, rebuild (recommended format)
    if isinstance(obj, dict):
        order = int(obj["order"])
        counts_raw = obj["counts"]
        overall_raw = obj["overall"]

        # Ensure values are Counters even if they were saved as dicts
        counts = {
            k: (v if isinstance(v, Counter) else Counter(v))
            for k, v in counts_raw.items()
        }
        overall = (
            overall_raw if isinstance(overall_raw, Counter) else Counter(overall_raw)
        )

        return NGramGRModel(order=order, counts=counts, overall=overall)

    raise TypeError(f"Unsupported model object in pkl: {type(obj)}")


# model = load_model("gr_predictor.pkl")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global model
    model = load_model("gr_predictor.pkl")
    yield


app = FastAPI(title="Trenball Predictor API", lifespan=lifespan)
# app = FastAPI(title="Trenball Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # don't use True with "*"
    allow_methods=["*"],
    allow_headers=["*"],
)


class NextRequest(BaseModel):
    current: str = ""


@app.get("/api/next/{cur}")
async def predict(cur):
    global _history

    if cur not in ("g", "r"):
        raise HTTPException(status_code=400, detail='current must be "g" or "r"')

    async with _lock:
        _history += cur
        _history = _history[-10:]
        pred, conf, probs = model.predict_next(_history)

        if pred == "g":
            print(_history, cur, pred, conf)
            return {"next": pred, "matched_pattern": conf}
        else:
            print(_history, cur, "-", conf)

    return {"next": "", "matched_pattern": None}


@app.get("/api")
def root():
    return {"message": "Trenball Predictor API is running 🚀"}


if __name__ == "__main__":
    # Recommended to run from CLI for reload:
    #   uvicorn app:app --host 0.0.0.0 --port 10081 --reload
    #   uvicorn app:app --port 10081
    uvicorn.run("app:app", port=10081, reload=False)
# host="0.0.0.0",
