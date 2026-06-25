import numpy as np
import tensorflow as tf

from models.dropout import make_dropout


def window_partition(x, window_size):
    batch, height, width, channels = tf.unstack(tf.shape(x))
    x = tf.reshape(x, (batch, height // window_size, window_size, width // window_size, window_size, channels))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    windows = tf.reshape(x, (-1, window_size * window_size, channels))
    return windows


def window_reverse(windows, window_size, height, width, channels):
    batch = tf.shape(windows)[0] // ((height // window_size) * (width // window_size))
    x = tf.reshape(windows, (batch, height // window_size, width // window_size, window_size, window_size, channels))
    x = tf.transpose(x, (0, 1, 3, 2, 4, 5))
    x = tf.reshape(x, (batch, height, width, channels))
    return x


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim, window_size, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = tf.keras.layers.Dense(dim * 3, use_bias=True)
        self.proj = tf.keras.layers.Dense(dim)

    def call(self, x):
        batch = tf.shape(x)[0]
        tokens = self.window_size * self.window_size
        qkv = self.qkv(x)
        qkv = tf.reshape(qkv, (batch, tokens, 3, self.num_heads, self.dim // self.num_heads))
        qkv = tf.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = tf.matmul(q * self.scale, k, transpose_b=True)
        attn = tf.nn.softmax(attn, axis=-1)
        out = tf.matmul(attn, v)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (batch, tokens, self.dim))
        return self.proj(out)


def swin_block(x, dim, window_size, num_heads, height, width):
    shortcut = x
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    x = tf.reshape(x, (-1, height, width, dim))
    pad_h = (window_size - height % window_size) % window_size
    pad_w = (window_size - width % window_size) % window_size
    x = tf.pad(x, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    padded_h = height + pad_h
    padded_w = width + pad_w
    windows = window_partition(x, window_size)
    attn = WindowAttention(dim, window_size, num_heads)(windows)
    x = window_reverse(attn, window_size, padded_h, padded_w, dim)
    x = x[:, :height, :width, :]
    x = tf.reshape(x, (-1, height * width, dim))
    x = tf.keras.layers.Add()([shortcut, x])
    shortcut = x
    y = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    y = tf.keras.layers.Dense(dim * 4, activation="gelu")(y)
    y = tf.keras.layers.Dense(dim)(y)
    x = tf.keras.layers.Add()([shortcut, y])
    return x


def patch_embed(x, embed_dim, patch_size):
    x = tf.keras.layers.Conv2D(embed_dim, patch_size, strides=patch_size, padding="same")(x)
    height = int(x.shape[1])
    width = int(x.shape[2])
    x = tf.keras.layers.Reshape((height * width, embed_dim))(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    return x, height, width


def patch_merging(x, height, width, dim):
    x = tf.reshape(x, (-1, height, width, dim))
    x0 = x[:, 0::2, 0::2, :]
    x1 = x[:, 1::2, 0::2, :]
    x2 = x[:, 0::2, 1::2, :]
    x3 = x[:, 1::2, 1::2, :]
    x = tf.concat([x0, x1, x2, x3], axis=-1)
    new_h = (height + 1) // 2
    new_w = (width + 1) // 2
    x = tf.reshape(x, (-1, new_h * new_w, 4 * dim))
    x = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
    x = tf.keras.layers.Dense(2 * dim, use_bias=False)(x)
    return x, new_h, new_w


def build_swin_encoder(inputs, embed_dim, depths, num_heads, window_size,
                       dropout_rate, encoder_dropout):
    x, height, width = patch_embed(inputs, embed_dim, patch_size=4)
    dim = embed_dim
    skips = []
    for stage, depth in enumerate(depths):
        for _ in range(depth):
            x = swin_block(x, dim, window_size, num_heads[stage], height, width)
        spatial = tf.reshape(x, (-1, height, width, dim))
        if encoder_dropout:
            spatial = make_dropout(dropout_rate, True)(spatial)
        skips.append(spatial)
        if stage < len(depths) - 1:
            x, height, width = patch_merging(x, height, width, dim)
            dim = dim * 2
    bottleneck = skips[-1]
    skip_list = list(reversed(skips[:-1]))
    return bottleneck, skip_list
