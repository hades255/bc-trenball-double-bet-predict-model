import json, re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

# --- CONFIG ---
HISTORY_WINDOW = 20
COLOR_MAP = {"red": 0, "green": 1}
MODEL_PATH = "trenball_lstm.h5"
DATA_FILE = "total.json"

# --- Step 1: Load and preprocess data ---
def load_data(filename):
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)
    return sorted(data, key=lambda x: int(x["roundId"]))

def preprocess(history):
    X, y = [], []
    for i in range(HISTORY_WINDOW, len(history)):
        window = history[i - HISTORY_WINDOW:i]
        label = COLOR_MAP.get(history[i]["color"])
        if label is None:
            continue

        seq = []
        for item in window:
            c = COLOR_MAP.get(item["color"], 0)
            m = 0.0
            match = re.search(r"(\d+(?:\.\d+)?)", item["multiplier"])
            if match:
                m = float(match.group(1))
            seq.append([c, m])
        X.append(seq)
        y.append(label)

    X = np.array(X)
    y = to_categorical(y, num_classes=2)
    return X, y

# --- Step 2: Build LSTM Model ---
def build_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(2, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model

# --- Step 3: Train Model ---
def train_model():
    history_data = load_data(DATA_FILE)
    X, y = preprocess(history_data)
    print(f"✅ Loaded {len(X)} samples for training.")

    model = build_model(input_shape=(HISTORY_WINDOW, 2))
    checkpoint = ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1)

    history = model.fit(
        X, y,
        epochs=40,
        batch_size=32,
        validation_split=0.1,
        callbacks=[checkpoint],
        verbose=1
    )

    # --- Plot Accuracy and Loss ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.legend()
    plt.title("Loss")
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

    print(f"✅ Training complete. Best model saved to {MODEL_PATH}")

# --- Step 4: Predict Next Color ---
def predict_next_color_lstm(history):
    from tensorflow.keras.models import load_model

    model = load_model(MODEL_PATH)
    if len(history) < HISTORY_WINDOW:
        raise ValueError("Not enough rounds to predict.")

    recent = history[-HISTORY_WINDOW:]
    seq = []
    for item in recent:
        c = COLOR_MAP.get(item["color"], 0)
        m = 0.0
        match = re.search(r"(\d+(?:\.\d+)?)", item["multiplier"])
        if match:
            m = float(match.group(1))
        seq.append([c, m])

    X = np.array([seq])
    probs = model.predict(X)[0]
    color = "green" if np.argmax(probs) == 1 else "red"
    # print(f"🎯 Prediction → {color.upper()} | Probabilities → red={probs[0]:.3f}, green={probs[1]:.3f}")
    return color, probs