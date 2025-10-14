import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import re

# --- Configurable Parameters ---
HISTORY_WINDOW = 20  # use last 20 rounds for prediction

# --- Color Encoding ---
COLOR_MAP = {"red": 0, "green": 1}


def preprocess_data(history):
    X, y = [], []

    # Make sure history is sorted by roundId ascending (oldest to newest)
    history = sorted(history, key=lambda x: int(x["roundId"]))

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


def train_model(history, model_path):
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
    print(classification_report(y_test, y_pred, target_names=["red", "green"]))

    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
