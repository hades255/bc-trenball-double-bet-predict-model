import pickle
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Tuple


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


# Load the old pkl (may contain a dataclass instance)
with open("gr_predictor.pkl", "rb") as f:
    m = pickle.load(f)

# Convert to a pure-serializable dict (no class dependency)
model_dict = {
    "order": int(m.order),
    "counts": {k: dict(v) for k, v in m.counts.items()},  # Counter -> dict
    "overall": dict(m.overall),                           # Counter -> dict
}

with open("gr_predictor.pkl", "wb") as f:
    pickle.dump(model_dict, f)

print("Converted gr_predictor.pkl to dict-based format ✅")
