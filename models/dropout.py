import tensorflow as tf


class MCDropout(tf.keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


def make_dropout(rate, active):
    if active and rate > 0.0:
        return MCDropout(rate)
    return tf.keras.layers.Activation("linear")
