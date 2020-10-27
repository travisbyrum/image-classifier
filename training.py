#!/usr/bin/env python

"""
Created October 27, 2020

@author: Travis Byrum
"""

import argparse
import os
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from typing import Any, Dict


DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180


def write_model(directory: str, model: Any, history):
    """Write model and training history to file."""

    model_json = model.to_json()

    with open(os.path.join(directory, "model.json"), "w") as json_file:
        json_file.write(model_json)

    model.save_weights(os.path.join(directory, "model.h5"))

    with open(os.path.join(directory, "history.json"), "w") as f:
        json.dump(history.history, f)


def main():
    """Entrypoint for training execution."""

    parser = argparse.ArgumentParser(description="Validate image detection model")
    parser.add_argument(
        "--data-dir", type=str, help="Data directory for model output", default="data"
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of training epochs", default=1
    )

    args = parser.parse_args()

    keras_dir = tf.keras.utils.get_file("flower_photos", origin=DATA_URL, untar=True)
    keras_dir = pathlib.Path(keras_dir)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        keras_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        keras_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
    )

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    normalization_layer = layers.experimental.preprocessing.Rescaling(1.0 / 255)
    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]

    model = Sequential(
        [
            layers.experimental.preprocessing.Rescaling(
                1.0 / 255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
            ),
            layers.Conv2D(16, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(5),
        ]
    )

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)
    write_model(args.data_dir, model, history)

    return model.summary()


if __name__ == "__main__":
    main()
