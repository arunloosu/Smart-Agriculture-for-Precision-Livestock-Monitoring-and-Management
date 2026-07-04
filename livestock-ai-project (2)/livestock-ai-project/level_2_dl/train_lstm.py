"""
Level DL (B) — LSTM behavior classification from raw accelerometer sequences
(captures temporal dependencies the ML-level hand-crafted features miss).

Dataset: same real accelerometer CSV used in level_1_ml (Beef Cattle Behavior
Dataset / ActBeCalf), fed in as raw un-windowed sequences.

Usage:
    python train_lstm.py --data ../level_1_ml/data/cattle_behavior.csv
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras import layers, models

SEQ_LEN = 50  # timesteps per sequence, matches WINDOW_SIZE in level_1_ml

COLUMN_MAP = {
    "x": ["acc_x", "x", "Ax", "accel_x"],
    "y": ["acc_y", "y", "Ay", "accel_y"],
    "z": ["acc_z", "z", "Az", "accel_z"],
    "label": ["behavior", "label", "activity", "class"],
}


def resolve_columns(df):
    resolved = {}
    for key, candidates in COLUMN_MAP.items():
        for c in candidates:
            if c in df.columns:
                resolved[key] = c
                break
        if key not in resolved:
            raise ValueError(f"Missing column for {key}; available: {list(df.columns)}")
    return resolved


def make_sequences(df, cols, seq_len=SEQ_LEN):
    x, y, z, label = df[cols["x"]].values, df[cols["y"]].values, df[cols["z"]].values, df[cols["label"]].values
    X_seq, y_seq = [], []
    for start in range(0, len(df) - seq_len, seq_len):
        sl = slice(start, start + seq_len)
        X_seq.append(np.stack([x[sl], y[sl], z[sl]], axis=1))  # (seq_len, 3)
        y_seq.append(pd.Series(label[sl]).mode()[0])
    return np.array(X_seq), np.array(y_seq)


def build_lstm(seq_len, n_features, n_classes):
    inputs = layers.Input(shape=(seq_len, n_features))
    x = layers.Masking()(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


def main(args):
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(args.data)
    cols = resolve_columns(df)

    X, y_raw = make_sequences(df, cols)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    n_samples, seq_len, n_features = X.shape
    X_flat = X.reshape(-1, n_features)
    scaler = StandardScaler().fit(X_flat)
    X = scaler.transform(X_flat).reshape(n_samples, seq_len, n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = build_lstm(seq_len, n_features, len(le.classes_))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("outputs/lstm_behavior_model.h5", save_best_only=True),
    ]
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=args.epochs,
                         batch_size=32, callbacks=callbacks)

    preds = np.argmax(model.predict(X_test), axis=1)
    report = classification_report(y_test, preds, target_names=le.classes_, output_dict=True, zero_division=0)
    with open("outputs/lstm_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    plt.figure()
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend(); plt.title("LSTM training curve")
    plt.savefig("outputs/training_curves_lstm.png", dpi=150)

    print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))
    print("Model saved to outputs/lstm_behavior_model.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    main(args)
