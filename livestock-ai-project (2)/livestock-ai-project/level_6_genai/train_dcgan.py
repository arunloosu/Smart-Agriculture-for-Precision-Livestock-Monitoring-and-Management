"""
Lightweight DCGAN for synthetic lumpy-skin-disease image augmentation.
CPU-feasible alternative to the diffusion pipeline (train_diffusion_lora.py).

Dataset: real "lumpy" class images from level_2_dl (Lumpy Skin Disease dataset).

Usage:
    python train_dcgan.py --data ../level_2_dl/data/images/lumpy --epochs 100
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

IMG_SIZE = 64
LATENT_DIM = 100
BATCH_SIZE = 32


def build_generator():
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 256, input_dim=LATENT_DIM),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, 4, strides=2, padding="same"),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(64, 4, strides=2, padding="same"),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(32, 4, strides=2, padding="same"),
        layers.BatchNormalization(), layers.LeakyReLU(0.2),
        layers.Conv2D(3, 5, padding="same", activation="tanh"),
    ])
    return model


def build_discriminator():
    model = tf.keras.Sequential([
        layers.Conv2D(64, 4, strides=2, padding="same", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(0.2), layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ])
    return model


def load_images(data_dir):
    ds = tf.keras.utils.image_dataset_from_directory(
        os.path.dirname(data_dir), labels=None, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    )
    ds = ds.map(lambda x: (x - 127.5) / 127.5)  # normalize to [-1, 1]
    return ds


def main(args):
    os.makedirs("outputs/dcgan_samples", exist_ok=True)
    ds = load_images(args.data)

    generator = build_generator()
    discriminator = build_discriminator()
    gen_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    disc_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    bce = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(real_images):
        batch = tf.shape(real_images)[0]
        noise = tf.random.normal([batch, LATENT_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_images = generator(noise, training=True)
            real_out = discriminator(real_images, training=True)
            fake_out = discriminator(fake_images, training=True)

            d_loss = bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)
            g_loss = bce(tf.ones_like(fake_out), fake_out)

        d_grads = disc_tape.gradient(d_loss, discriminator.trainable_variables)
        g_grads = gen_tape.gradient(g_loss, generator.trainable_variables)
        disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))
        gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
        return d_loss, g_loss

    for epoch in range(args.epochs):
        d_losses, g_losses = [], []
        for batch in ds:
            d_l, g_l = train_step(batch)
            d_losses.append(float(d_l)); g_losses.append(float(g_l))
        print(f"Epoch {epoch+1}/{args.epochs} — d_loss={np.mean(d_losses):.3f} g_loss={np.mean(g_losses):.3f}")

        if (epoch + 1) % 10 == 0:
            noise = tf.random.normal([16, LATENT_DIM])
            samples = (generator(noise, training=False) * 127.5 + 127.5).numpy().astype("uint8")
            for i, img in enumerate(samples):
                tf.keras.utils.save_img(f"outputs/dcgan_samples/synthetic_e{epoch+1}_{i}.png", img)

    generator.save("outputs/dcgan_generator.h5")
    print("Saved generator to outputs/dcgan_generator.h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to real 'lumpy' image class folder")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    main(args)
