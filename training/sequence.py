import numpy as np
import tensorflow as tf

from config import BATCH_SIZE, NUM_CLASSES
from training.augment import augment_pair


class SliceSequence(tf.keras.utils.Sequence):
    def __init__(self, images, labels, batch_size=BATCH_SIZE, augment=False, seed=0, shuffle=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(images))
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, batch_index):
        start = batch_index * self.batch_size
        end = start + self.batch_size
        batch_idx = self.indices[start:end]
        images = self.images[batch_idx].copy()
        labels = self.labels[batch_idx].copy()
        if self.augment:
            for i in range(len(images)):
                images[i], labels[i] = augment_pair(images[i], labels[i], self.rng)
        targets = np.eye(NUM_CLASSES, dtype=np.float32)[labels]
        return images, targets

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
