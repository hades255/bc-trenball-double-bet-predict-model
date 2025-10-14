from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import uvicorn
import json
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import threading

# from lstm_model import predict_next_color_lstm

# Load your model (trained and saved as trenball_model.pkl)
MODEL_PATH = ["trenball_model.pkl", "trenball_model_.pkl"]
MODEL_PATH_INDEX = 0
FINAL_FILE = "final.json"


HISTORY_WINDOW = 20
COLOR_MAP = {"red": 0, "green": 1}
REVERSE_COLOR_MAP = {0: "red", 1: "green"}

COUNT = 0

app = FastAPI(title="Trenball Predictor API")

# Optional: enable CORS so your Chrome extension can call it
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify: ["chrome-extension://your-extension-id"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Utility function ---
def predict_next_color(history):
    if len(history) < HISTORY_WINDOW:
        raise ValueError("Not enough rounds in history")

    # Take last 20 rounds
    window = history[-HISTORY_WINDOW:]
    features = []
    for item in window:
        color = COLOR_MAP.get(item.get("color"), 0)
        try:
            multiplier = float(item.get("multiplier", "1.0").replace("×", ""))
        except Exception:
            multiplier = 1.0
        features.append(color)
        features.append(multiplier)

    # Predict
    model_path=MODEL_PATH[MODEL_PATH_INDEX]
    model = joblib.load(model_path)
    pred = model.predict([features])[0]
    prob = model.predict_proba([features])[0]
    return REVERSE_COLOR_MAP[pred], prob.tolist()


def merge_and_save_history(new_history):
    """Merge new history into final.json, deduplicate and sort."""
    if os.path.exists(FINAL_FILE):
        with open(FINAL_FILE, "r", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # Combine and deduplicate by roundId
    combined = {str(item["roundId"]): item for item in existing + new_history}
    merged_list = list(combined.values())

    # Sort by roundId (as int if numeric)
    merged_list.sort(key=lambda x: int(x.get("roundId", 0)))
    
    merged_list = merged_list[-1500:]

    # Save back
    with open(FINAL_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2, ensure_ascii=False)

    print(f"✅ Merged and saved {len(merged_list)} total items to {FINAL_FILE}")
    return merged_list


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


def run_train_model(history, model_path):
    try:
        merged_list = merge_and_save_history(history)
        train_model(merged_list, model_path)
        print("Model training completed in background")
    except Exception as e:
        print(f"Error in background training: {e}")

# --- Endpoint ---
@app.post("/api/predict")
async def predict(request: Request):
    global COUNT
    global MODEL_PATH_INDEX
    body = await request.json()
    history = body.get("history", [])
    if not history:
        return {"error": "No history provided"}

    COUNT += 1

    try:
        if COUNT >= 5:
            print("recreate model")
            COUNT = 0
            model_path = MODEL_PATH[MODEL_PATH_INDEX]
            MODEL_PATH_INDEX = (MODEL_PATH_INDEX + 1) % 2
            # merged_list = merge_and_save_history(history)
            # train_model(merged_list, MODEL_PATH[MODEL_PATH_INDEX])
            
            thread = threading.Thread(
                target=run_train_model,
                args=(history, model_path),
                daemon=True
            )
            thread.start()

        color, prob = predict_next_color(history)
        # color, prob = predict_next_color_lstm(history)
        print(
            f"🎯 Predicted Next Color: {color} (Probabilities: red={prob[0]:.2f}, green={prob[1]:.2f}"
        )
        result = {"color": color}
        # result = {"color": color}

        return result
    except Exception as e:
        return {"error": str(e)}


# --- Example GET test route ---
@app.get("/api")
def root():
    return {"message": "Trenball Predictor API is running 🚀"}


# --- Run server ---
if __name__ == "__main__":
    # uvicorn.run("app:app", host="0.0.0.0", port=5001, reload=True)
    uvicorn.run("app:app", port=5001, reload=True)
