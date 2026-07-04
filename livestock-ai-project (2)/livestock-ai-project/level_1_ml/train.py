"""
Level ML — Baseline cattle-behavior classifier from accelerometer data.

Dataset: Beef Cattle Behavior Dataset (Kaggle: lucyfirst/beef-cattle-behavior-data-set)
or any real tri-axial accelerometer + behavior-label export (e.g. ActBeCalf).

Usage:
    python train.py --data ./data/cattle_behavior.csv
"""
import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# Real-world exports use varying column names — adjust here if needed.
COLUMN_MAP = {
    "x": ["acc_x", "x", "Ax", "accel_x"],
    "y": ["acc_y", "y", "Ay", "accel_y"],
    "z": ["acc_z", "z", "Az", "accel_z"],
    "label": ["behavior", "label", "activity", "class"],
}
WINDOW_SIZE = 50  # samples per window (~5s at 10Hz — adjust to your file's sample rate)


def resolve_columns(df: pd.DataFrame) -> dict:
    resolved = {}
    for key, candidates in COLUMN_MAP.items():
        for c in candidates:
            if c in df.columns:
                resolved[key] = c
                break
        if key not in resolved:
            raise ValueError(
                f"Could not find a column for '{key}'. "
                f"Available columns: {list(df.columns)}. Update COLUMN_MAP."
            )
    return resolved


def window_features(df: pd.DataFrame, cols: dict, window: int = WINDOW_SIZE) -> pd.DataFrame:
    rows = []
    x, y, z, label = df[cols["x"]].values, df[cols["y"]].values, df[cols["z"]].values, df[cols["label"]].values
    for start in range(0, len(df) - window, window):
        sl = slice(start, start + window)
        wx, wy, wz = x[sl], y[sl], z[sl]
        window_label = pd.Series(label[sl]).mode()[0]  # majority label in window
        feats = {
            "mean_x": wx.mean(), "mean_y": wy.mean(), "mean_z": wz.mean(),
            "std_x": wx.std(), "std_y": wy.std(), "std_z": wz.std(),
            "min_x": wx.min(), "min_y": wy.min(), "min_z": wz.min(),
            "max_x": wx.max(), "max_y": wy.max(), "max_z": wz.max(),
            "rms_x": np.sqrt(np.mean(wx ** 2)), "rms_y": np.sqrt(np.mean(wy ** 2)), "rms_z": np.sqrt(np.mean(wz ** 2)),
            "zcr_x": ((wx[:-1] * wx[1:]) < 0).sum() / window,
            "corr_xy": np.corrcoef(wx, wy)[0, 1] if wx.std() > 0 and wy.std() > 0 else 0.0,
            "corr_xz": np.corrcoef(wx, wz)[0, 1] if wx.std() > 0 and wz.std() > 0 else 0.0,
            "label": window_label,
        }
        rows.append(feats)
    return pd.DataFrame(rows).dropna()


def main(args):
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(args.data)
    cols = resolve_columns(df)
    print(f"Resolved columns: {cols}")

    feat_df = window_features(df, cols)
    print(f"Built {len(feat_df)} windows across {feat_df['label'].nunique()} behaviors: "
          f"{feat_df['label'].unique().tolist()}")

    le = LabelEncoder()
    y = le.fit_transform(feat_df["label"])
    X = feat_df.drop(columns=["label"]).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    class_counts = pd.Series(y).value_counts().sort_index()
    stratify = y if class_counts.min() >= 2 else None
    if stratify is None:
        print(
            "Warning: some classes have too few windows for stratified splitting; "
            f"class counts={class_counts.to_dict()}. Falling back to an unstratified split."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=stratify, random_state=42
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=42),
    }

    metrics = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        metrics[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
            "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
            "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
            "roc_auc_ovr": roc_auc_score(y_test, proba, multi_class="ovr"),
        }
        print(f"\n=== {name} ===")
        print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))
        joblib.dump(model, f"outputs/model_{name}.pkl")

    joblib.dump(scaler, "outputs/scaler.pkl")
    joblib.dump(le, "outputs/label_encoder.pkl")

    with open("outputs/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix for the best model by F1
    best_name = max(metrics, key=lambda k: metrics[k]["f1_macro"])
    best_model = joblib.load(f"outputs/model_{best_name}.pkl")
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_estimator(
        best_model, X_test, y_test, display_labels=le.classes_, ax=ax, xticks_rotation=45
    )
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png", dpi=150)

    if best_name == "random_forest":
        importances = best_model.feature_importances_
        feat_names = feat_df.drop(columns=["label"]).columns
        order = np.argsort(importances)[::-1]
        plt.figure(figsize=(8, 5))
        sns.barplot(x=importances[order], y=np.array(feat_names)[order])
        plt.title("Random Forest Feature Importance")
        plt.tight_layout()
        plt.savefig("outputs/feature_importance.png", dpi=150)

    print(f"\nBest model: {best_name} (F1-macro={metrics[best_name]['f1_macro']:.3f})")
    print("Artifacts written to ./outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_data = os.path.join(os.path.dirname(__file__), "data", "cattle_behavior.csv")
    parser.add_argument(
        "--data",
        default=default_data,
        help="Path to downloaded accelerometer CSV. Defaults to level_1_ml/data/cattle_behavior.csv",
    )
    args = parser.parse_args()

    if not os.path.exists(args.data):
        parser.error(
            f"Data file not found: {args.data}\n"
            "Please place the dataset at the default location or pass --data /path/to/file"
        )

    main(args)
