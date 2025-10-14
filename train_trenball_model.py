import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re

# --- Configurable Parameters ---
HISTORY_WINDOW = 20  # use last 20 rounds for prediction
MODEL_FILENAME = "trenball_model_.pkl"
INPUT_JSON_FILE="total.json"
JSON_FILE="final.json"

# --- Color Encoding ---
COLOR_MAP = {"moon": 0, "green": 1}
REVERSE_COLOR_MAP = {0: "moon", 1: "green"}


def preprocess_data(history):
    X, y = [], []

    # Make sure history is sorted by roundId ascending (oldest to newest)
    history = sorted(history, key=lambda x: int(x["roundId"]))
    
    for item in history:
        if item.get("color") == "moon":
            item["color"] = "green"
    
    with open(JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    for i in range(HISTORY_WINDOW, len(history) - 1):
        window = history[i - HISTORY_WINDOW : i]
        next_color = history[i]["color"]

        # Skip if any missing color
        if not all(item["color"] in COLOR_MAP for item in window + [history[i]]):
            continue

        # Feature vector = [color_0, mult_0, color_1, mult_1, ..., color_19, mult_19]
        features = []
        for item in window:
            features.append(COLOR_MAP[item["color"]])
            multiplier = 0.0
            match = re.search(r"(\d+(?:\.\d+)?)", item["multiplier"])
            if match:
                multiplier = float(match.group(1))
            features.append(multiplier)

        X.append(features)
        y.append(COLOR_MAP[next_color])

    return np.array(X), np.array(y)


def train_model(history):
    X, y = preprocess_data(history)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("📊 Evaluation:")
    print(classification_report(y_test, y_pred, target_names=["red", "green"], zero_division=0))

    joblib.dump(model, MODEL_FILENAME)
    print(f"✅ Model saved to {MODEL_FILENAME}")


def predict_next_color(latest_history):
    if len(latest_history) < HISTORY_WINDOW:
        raise ValueError("Not enough history to predict")

    model = joblib.load(MODEL_FILENAME)

    # Prepare feature from last 20
    window = latest_history[-HISTORY_WINDOW:]
    features = []
    for item in window:
        features.append(COLOR_MAP.get(item["color"], 0))
        multiplier = 0.0
        match = re.search(r"(\d+(?:\.\d+)?)", item["multiplier"])
        if match:
            multiplier = float(match.group(1))
        features.append(multiplier)

    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0]

    return REVERSE_COLOR_MAP[pred], prob.tolist()


# --- Example usage ---
if __name__ == "__main__":
    # Load your JSON array of Trenball history (replace with your file or data source)
    with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
        history = json.load(f)

    train_model(history)

    # Example prediction:
    color, prob = predict_next_color(history[-HISTORY_WINDOW:])
    print(
        f"🎯 Predicted Next Color: {color} (Probabilities: red={prob[0]:.2f}, green={prob[1]:.2f})"
    )
