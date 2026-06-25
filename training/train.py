import os
import numpy as np
import tensorflow as tf

from config import (
    EPOCHS,
    LEARNING_RATE,
    LR_REDUCE_FACTOR,
    LR_REDUCE_PATIENCE,
    LR_FLOOR,
    BATCH_SIZE,
    DROPOUT_RATE,
    DROPOUT_PLACEMENT,
)
from models import build_model
from training.losses import composite_loss
from training.sequence import SliceSequence


def compile_model(model):
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=composite_loss)
    return model


def build_callbacks(checkpoint_path):
    return [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=LR_REDUCE_FACTOR,
            patience=LR_REDUCE_PATIENCE,
            min_lr=LR_FLOOR,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]


def train_model(backbone, train_images, train_labels, val_images, val_labels,
                checkpoint_path, dropout_active=True, dropout_rate=DROPOUT_RATE,
                placement=DROPOUT_PLACEMENT, seed=0, epochs=EPOCHS):
    tf.keras.utils.set_random_seed(seed)
    model = build_model(
        backbone, dropout_active=dropout_active,
        dropout_rate=dropout_rate, placement=placement
    )
    compile_model(model)
    train_seq = SliceSequence(train_images, train_labels, BATCH_SIZE, augment=True, seed=seed)
    val_seq = SliceSequence(val_images, val_labels, BATCH_SIZE, augment=False, seed=seed, shuffle=False)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    callbacks = build_callbacks(checkpoint_path)
    history = model.fit(
        train_seq,
        validation_data=val_seq,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    model.load_weights(checkpoint_path)
    return model, history.history
