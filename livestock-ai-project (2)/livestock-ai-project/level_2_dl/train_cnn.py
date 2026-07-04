"""
Level DL (A) — CNN health assessment: healthy vs. Lumpy Skin Disease cattle images.
Transfer learning on MobileNetV2 (ImageNet weights).

Dataset: Kaggle warcoder/lumpy-skin-images-dataset
Expected layout: ./data/images/{healthy,lumpy}/*.jpg

Usage:
    python train_cnn.py --data ./data/images --epochs 15
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def build_model(num_classes: int) -> tf.keras.Model:
    base = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
    base.trainable = False  # freeze for transfer learning; unfreeze last layers to fine-tune

    inputs = layers.Input(shape=IMG_SIZE + (3,))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(1 if num_classes == 2 else num_classes,
                            activation="sigmoid" if num_classes == 2 else "softmax")(x)
    return models.Model(inputs, outputs)


def main(args):
    os.makedirs("outputs", exist_ok=True)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="training", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data, validation_split=0.2, subset="validation", seed=42,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")

    # Light augmentation — real cattle-disease research shows this improves
    # generalization given the modest dataset size (~800-8000 images).
    augment = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])
    train_ds = train_ds.map(lambda x, y: (augment(x, training=True), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    model = build_model(len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy" if len(class_names) == 2 else "sparse_categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("outputs/cnn_lumpy_detector.h5", save_best_only=True),
    ]

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    # Optional fine-tuning: unfreeze top layers of the backbone for a few epochs
    base = model.layers[2]
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss=model.loss, metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    history_ft = model.fit(train_ds, validation_data=val_ds, epochs=5, callbacks=callbacks)

    val_metrics = model.evaluate(val_ds, return_dict=True)
    with open("outputs/cnn_metrics.json", "w") as f:
        json.dump(val_metrics, f, indent=2)

    for h, tag in [(history, "frozen"), (history_ft, "finetune")]:
        plt.figure()
        plt.plot(h.history["accuracy"], label="train_acc")
        plt.plot(h.history["val_accuracy"], label="val_acc")
        plt.legend(); plt.title(f"CNN training curve ({tag})")
        plt.savefig(f"outputs/training_curves_cnn_{tag}.png", dpi=150)

    print("Final validation metrics:", val_metrics)
    print("Model saved to outputs/cnn_lumpy_detector.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to ./data/images with class subfolders")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    main(args)
