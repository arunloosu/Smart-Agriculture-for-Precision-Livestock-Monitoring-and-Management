"""
Evaluate the real value of synthetic augmentation:
1. Downstream utility — does adding synthetic images improve the Level-DL
   CNN's accuracy on a held-out REAL test set?
2. Robustness — does the augmented model hold up better under perturbations
   (brightness/blur/contrast) that simulate diverse field conditions?

Usage:
    python stress_test.py --cnn ../level_2_dl/outputs/cnn_lumpy_detector.h5 \
        --synthetic ./outputs/synthetic_images --real_test ../level_2_dl/data/images_test
"""
import argparse
import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = (224, 224)


def perturb(ds):
    aug = tf.keras.Sequential([
        layers.RandomBrightness(0.3),
        layers.RandomContrast(0.3),
        layers.GaussianNoise(0.05),
    ])
    return ds.map(lambda x, y: (aug(x, training=True), y))


def main(args):
    os.makedirs("outputs", exist_ok=True)
    real_test = tf.keras.utils.image_dataset_from_directory(
        args.real_test, image_size=IMG_SIZE, batch_size=32, shuffle=False
    )

    baseline_model = tf.keras.models.load_model(args.cnn)
    baseline_metrics = baseline_model.evaluate(real_test, return_dict=True)

    perturbed_test = perturb(real_test)
    baseline_robust_metrics = baseline_model.evaluate(perturbed_test, return_dict=True)

    result = {
        "baseline_on_clean_real_test": baseline_metrics,
        "baseline_on_perturbed_real_test": baseline_robust_metrics,
        "note": (
            "To complete the comparison, retrain the CNN (level_2_dl/train_cnn.py) "
            "on real+synthetic combined data, save as cnn_lumpy_detector_augmented.h5, "
            "then re-run this script pointing --cnn at that file to fill in "
            "'augmented_on_clean_real_test' / 'augmented_on_perturbed_real_test'."
        ),
    }
    with open("outputs/robustness_report.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cnn", required=True, help="Path to a trained CNN .h5 (baseline or augmented)")
    parser.add_argument("--synthetic", required=True)
    parser.add_argument("--real_test", required=True, help="Held-out REAL test split, untouched by training")
    args = parser.parse_args()
    main(args)
