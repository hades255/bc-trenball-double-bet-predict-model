import os
import pickle
from collections import Counter, defaultdict
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


def train_ngram_model(training: str, order: int = 5) -> NGramGRModel:
    s = _clean_gr(training)
    counts: Dict[str, Counter] = defaultdict(Counter)
    overall = Counter()

    # Build counts for contexts of length 0..order
    for i in range(len(s) - 1):
        nxt = s[i + 1]
        overall[nxt] += 1

        # for each context length k, add an example
        for k in range(order + 1):
            ctx_start = i - k + 1
            if k == 0:
                ctx = ""
            else:
                if ctx_start < 0:
                    continue
                ctx = s[ctx_start : i + 1]
            counts[ctx][nxt] += 1

    return NGramGRModel(order=order, counts=dict(counts), overall=overall)


def save_model(model: NGramGRModel, path: str = "gr_predictor.pkl") -> None:
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str = "gr_predictor.pkl") -> NGramGRModel:
    with open(path, "rb") as f:
        return pickle.load(f)


def read_string(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r+b") as f:
        data = f.read()

    text = data.decode("utf-8", errors="ignore")
    text = "".join(ch for ch in text.lower() if ch in ("g", "r"))
    return text


# ------------------ Example usage ------------------
if __name__ == "__main__":
    
    for i in range(4, 12):
        order = i

        # training_string = read_string("gr_limbo.txt")
        training_string = read_string("trenball.txt")
        model = train_ngram_model(training_string, order=order)  # try 4..8
        save_model(model, "gr_predictor.pkl")

        text = "rrrrrrgggrggrrrgggrrgrrrrggrrrgrrrrrggrrgrggrrgrgrgrrgrgrrgggrgrrrrrrrrrrgrrrrggrgrrgggggggggggggggrgggggggrgrrgrgggrgrrgrrrrggrgrggrrgrrgrrgrgrgrgggggrgrrrgrggrggrggrgrgrggrggggrgrgrrgrggrrggrrrgrgrgrrrrggrgggrggrrrrggggrrggrgggrgrrgrrrggrgrrrrggrgrggggrrggggrrrrggrgggrrggrrrgggrgrggrrgrggggrgrggrgggrgrggrgrgrggrrrrggrrgrrrggrrrrrrgrgrrrgggrrgrgrgrrgrrgggrrrrggggrrgggrgrgggggrrgrgrrrrggggggggrrrrrrgrggrgrrrgrgggrgggrgrgggrgrgggrrrgrggggggrgrrrgrrrrrggrrggrrgrgrgrggrrgrgrgrrggrrggrrgrrgrgrgggggrrrgrrrrrrggrggrrrrggggrrgggrggrgrrgrrrrgrggrggrrrrrrgrrrggrrggrrrrrgrrgggrrggrgrrrrrggrrgrrrgrgggggrgrrrgrggrgrrrggrrrggrrrrggrrrrrrgrrrrgrrgrgrggrggrrgrggrgrrrgrggrggrgrgggrgrrrgrrggggrrrrrrrgrgggrggggrgrgrrrrrggrrgrrggrrrrggrrgrgggggggggrrrgggggrggrrgrrgggrgrgggrrgggggrrgrrgrggrrgrggggrrgrgrrrggggggrggrrgrggrgrrrrggrgggrgggrgrrgggrrggrgrrrrggggggggrrrrrrggrrrrrrgrrrggrrgrgrrgggrrggrrggrgrrrrgrrrrrggrggrgrrrgggggrggrrgggrrrgrgrggrrrgrrrrrrggrrrrrrrgrrrgrgrgrgrgrgggrrrggrggrgrggrrrrrrggrggrgggrrgrrrgrrgggrrrrgggrggrggrrrrrggrgggggrrrrggrgrrrgrgrrggggrrgggrrgrrrgrgrrrrrrgrrgrggrrrgrrgrrrrgrrrggrrrrrrgggrgrggrgggrrrrrgrgrgrrrrggrrrrrgggrrrgrrrrrrggrrgrgrrgggrrgrrrggrrrrrrrrrrrgggrrrrggrgrrrrggrgrrgrgrrgrrgrggrgrgrrrrggrgrgggggrgggrrrgrrggrggrgrgrggrrrgrggrrggrrggrggrrgrrggrgggrgggrrrrrrgrgrgrrrggrgrgrrggrrrgrrrggrggrgrgrggrggggggrgrrrrrggggrrrgrggrrgggrrrrrrgrggrrgrgggggrgrgrrrggrggrgggrrrrrgrrgrgrggrrgggrrgrgrrrrrrrrrgrggrrgrrrrrgrgrggrrrgrgrgrgrgrgggggrrrgggrgrrggggggggrrgrrggrrrgrrgrrrrgrrgrggggrrgrggrgrgrgrggrrgrggggggggrgrgggggrrrrrrgrrrrrrrrrggrgrrggrgggggrrrrgggrgrrggrrrgrrggrggrggrrgrrrrgrrgrrrrggrgrrrrrrrrggggggrgrgrgrgrgrggrgggrggrggggrggrrgrgggrggrggrrgrgggrrgrrgrrgrrggrrrrrrrrrgrggrgrgggggrrrrrggrrgrgrrgggrgrrgrrggrrrgggggrggrrgrrrgggrgrrggrgrrgggrgrrggrrrrggrrggggrgrgrrgrgggrrggrrrrgrgggrrrgggggrrgrrgggrgrrgrggrrgrggrgrgrggrrrrrrgggrrrggrrgrgrrgrgggrggrrgrrrgrrggggrrrrgggrgrgrrrrrrrggrggrgggrgrrrrrgrrrrrrrgggrggrrgrgrrggrggggrrggrgggrgrgggrrgrgrrggggrgggrrrrrrgggrgrggrgrggrrrrgrgrgrrgrgrggrrrggrrggrrrggrggggggrrrrggggrggrggrggggrggggggrrggrrrrrrrggrgrrrgrrrrgrgrggrrgrrrgggrrgrggrrrrgrrrgrgrrggggggggrggrrgrrrggrrgrrrrgggrrgrgrrggrgggrgrrggrrrggggrrgrrgggrggggrrgrrgrggrrggrggrrggggrrggrrggggggrgggrgrrgrrrgrggrgrrgrrrrrgrrggrrrrgggrrggrgrrrrrrrgrrrggggggggrrrrrgrrgrgrrrggrrrrrrgrrgrgrrggrrrrrgrrgrrrrgrrrggrggrrgrrggrgrrgrrggrrgrrrgggrgggrgrgggrggrggrgrgrggrgrgrgrgggrgrrgrgrggrgrggrrrgrrrgrgrggrgrrrrrrrrggrgrrrgrgrrgrrgrrgrgggggrgrrrrgrgrgrggrrrgggrgrrggrgggrrgrrrgrgrgrrrgrggrgggrgggrgrrrrrrgrrggggggrrgrggrrrgrrgrrgrggrrggrggrrgggrggrrggrrrggrggrggggrrrgrrggggrrrgrrrggggrgrrggggrgrgrgrrrrgrrrrrgrgrgrrrrggrrrggggrgrrgrggrgggggrgrgrgrrrgggrgrgrrrrrgrgrrgrgrrrrgggggrgrrggrrgrgrrgrggrrrggggrggrrrgggrrrrgrgrgrggrgrrrggrrrgrgggrgrggrgggggrrrrrrgggrrrgrrrrrrgrrgrrrrgrgggrgrggrrgrrgggrgggrggrgrgrrrggrggrrrrrggrrgrgggrgrrgrrgggggggrrgrgrgrggrggrgrggggrrrrrgrggrgrrgrrrrrgrrgrrggrggrrggrrggrggggggrrrrgggrggrgrrrrrgggrggrggrgrgrgggrgrrrggrgrgrgrrgrggrrggrgrrrggggrrrgrrggrrrrrrggrrgggggrgrrgrggrggrrrrgrgrrgrggrrrrrggggggrggggrrgggrgggrrgrrrrrrrrrrrgrggrrrggrrggrgrggggrrgrgggrrgggggggggrggggggrgrgrrggrgrgrgggrrgrrgrrgrrrrrgrggrrrrrrrrggrrgggrrggrrrrgrgrrrgggggrggrgrgrrgrgrrgrggrgrgrgggrrrggggrgrrrgrgrrrggrgrrgrrgrgggggrrgggrggggrrggrrgrggrgrgrrgrrgrggrrrgggrggrrrrgrrrgrgrgrgggrrggrgrggggrrgrrrgrgrrrggrgrggggrrrgrgrgrrggrrrrrrrrgrgrgrggrggrgrrrrggrrgrgrgrgrrrgggrrrrrrrrrrgrrrrgrggrrrgggrrgrrggggrrrgrgrgggrrrrgrgrrgrrrrgrgrrrrgrrggrgrrrgggrgggrgrrrggrggrgrrrgrrrrrgggrggggrrrrgrrrgggrrrrgrrgggggggrgrrrgrgggrrgrrgggrrrrrgrrrrgrgrgrgrgggrgrgrgrrgrggrrgrrgrrgrrrggrrgrgrgggrgrgrggggggrggrgggrrgrrrgrgggrgrrrrgrrgrgrrrgrggrggggrgrrgrgggrrrrrggrrrggrgrgrgrrrrgggrrgrgrrggrrrggggrggrrrgrrrggrgggrgggrggrgrgrgggrrgrrrrgrrrgrgggrrgrggrrrrrgggggrggrggrggggrrggrgrgrrggggrggrgrrgggrggggrrrgrrggrrrrrggrrggrrrrrrrggrrgrrgrgggrrggggrrrrrrggrrggrgggggrrgrgrrggrrrgrgrrrrrrgggrrggrrrrggrrgggrrggrggrgrggrgrggrggrrrrrggggrgrrrgrrgggrgggggrrgrrgggrgrggggrgrrgrgrgrgrgrgrrgrrgggggrggrggrgrrrrgrgrgrgrgggggrggggrgrgrgggrgrrrgrrrgggggrrggrrrrgrggrggrrrggggggggrgrrrgrgrgrrgggrgggrgrrggggrrrrgrrrrrggggrgrrgrggrgggrrrrrrgrggrrgrrgrggggrgrrrrrrggrrrrgrrrrgrgrrggrrrrrrgrgrrrrgrgrrggrrggggrrrggrggrggrgrrrrrgrrggrgrrgrgrgrrrgrgrggrrggrggrggrrggrrrrgrggrrrrgrrrgrrgrrgggrgrrrrrggggrrrgggrggrggrrgrrrggggrgrrrrrrgrggggggggrrrrgrggrrrrrgrggrrgrrrggrrrgrrggrggrggrrrrrggrggggrggggrggrrrrrrgggrrggrrrrrggggggrgrrgrrrrrgggrrggggrrgrgrgrrggrrgrrgrgrgrrggrrgrgggrrrgrrgrgggrrgrggggrgrrrgrrrrrgrrggrrrggggrgrrrrggggrgrgggggrrgggrggrgrrrrrrrggrgrrgrgggrgrrrrgrggrgrrgrgrrgggrrgrggrgggrgrgrgrgggrrgggrrgrrgggrrrggrgggrrggrrrrrgggggggggrgggrgrrggggrgrrggrrgrrgrggrgrgrgrggggrggrgggrggggrrrgrrrrggrgrrrrggrggrrrrggrrrggrrgrrggrgggrggrgggrgrrggrgrgrrrrrrgrrgggrrrrrrrrrgrgggrrrrgggggggrrrrgrrggrgrgrrggrggrgrgggrgggggrggrgggrrrggrgrgrggrggrgrrrrrggrrrgrrgrgggrggggrgrrggrrrrrrggrrgrrggggrggrgggrrgrrgrrgrgrgrgrggrrrrggrgrrrggrgrgrrrrgrgrrrrrgrggggggrrrgrgrgrrrggrrrgrggrrggrrrgrggggrggrrggrrgggrgggrgggrgrrrgrggggrgggggggrrgrrrggrgrgrgggrrrggrrgggrrrrrrggrrgggrggrggrrrgrgggrrrgrggrrrgggrggrgrgrrrgrggggrrgrrrrrgrrrggrgggggrggggggrgggrggrgrrgrrrrrrggggrgrrrrrrgrggrrrrgrrrgrrggggrgrgrgrrrgggggrgrgggrrrgrrrrggrgrgrrgggg"

        model2 = load_model("gr_predictor.pkl")

        # history = "rgrrgrgrrggrrrrgggrrg"
        # pred, conf, probs = model2.predict_next(history)
        # print("history:", history)
        # print("pred:", pred, "confidence:", round(conf, 4), "probs:", probs)

        rwin = 0
        gwin = 0
        rlose = 0
        glose = 0
        for i in range(20, len(text) - 1):
            history = text[i - 20 : i]
            pred, conf, probs = model2.predict_next(history)
            # print("history:", history)
            cur = text[i - 1]
            next_cur = text[i]
            # print(
            #     i,
            #     "cur:",
            #     cur,
            #     "next:",
            #     next_cur,
            #     "pred:",
            #     pred,
            #     # "probs:",
            #     # probs,
            #     "matched:",
            #     next_cur == pred,
            #     "confidence:",
            #     round(conf, 4),
            # )
            if pred == "g":
                if next_cur == pred:
                    gwin += 1
                else:
                    glose += 1
            else:
                if next_cur == pred:
                    rwin += 1
                else:
                    rlose += 1

        print(order, "gw:", gwin, "gl:", glose, "rw:", rwin, "rl:", rlose)
