import tensorflow as tf

from models.dropout import make_dropout


def conv_bn_swish(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("swish")(x)
    return x


def inverted_residual(x, out_filters, strides, expansion):
    in_filters = int(x.shape[-1])
    hidden = in_filters * expansion
    shortcut = x
    y = conv_bn_swish(x, hidden, kernel_size=1)
    y = tf.keras.layers.DepthwiseConv2D(3, strides=strides, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("swish")(y)
    y = tf.keras.layers.Conv2D(out_filters, 1, padding="same", use_bias=False)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    if strides == 1 and in_filters == out_filters:
        y = tf.keras.layers.Add()([shortcut, y])
    return y


def transformer_encoder(x, dim, num_heads, mlp_dim, depth):
    for _ in range(depth):
        shortcut = x
        y = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        y = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)(y, y)
        x = tf.keras.layers.Add()([shortcut, y])
        shortcut = x
        y = tf.keras.layers.LayerNormalization(epsilon=1e-5)(x)
        y = tf.keras.layers.Dense(mlp_dim, activation="swish")(y)
        y = tf.keras.layers.Dense(dim)(y)
        x = tf.keras.layers.Add()([shortcut, y])
    return x


def mobilevit_block(x, transformer_dim, num_heads, mlp_dim, depth, patch_size):
    in_channels = int(x.shape[-1])
    height = int(x.shape[1])
    width = int(x.shape[2])
    local = conv_bn_swish(x, in_channels, kernel_size=3)
    local = tf.keras.layers.Conv2D(transformer_dim, 1, padding="same")(local)
    ph, pw = patch_size, patch_size
    pad_h = (ph - height % ph) % ph
    pad_w = (pw - width % pw) % pw
    local = tf.pad(local, [[0, 0], [0, pad_h], [0, pad_w], [0, 0]])
    new_h = height + pad_h
    new_w = width + pad_w
    num_patches = (new_h // ph) * (new_w // pw)
    patch_area = ph * pw
    folded = tf.reshape(local, (-1, new_h // ph, ph, new_w // pw, pw, transformer_dim))
    folded = tf.transpose(folded, (0, 2, 4, 1, 3, 5))
    folded = tf.reshape(folded, (-1, patch_area, num_patches, transformer_dim))
    folded = tf.reshape(folded, (-1, num_patches, transformer_dim))
    transformed = transformer_encoder(folded, transformer_dim, num_heads, mlp_dim, depth)
    transformed = tf.reshape(transformed, (-1, patch_area, num_patches, transformer_dim))
    unfolded = tf.reshape(transformed, (-1, ph, pw, new_h // ph, new_w // pw, transformer_dim))
    unfolded = tf.transpose(unfolded, (0, 3, 1, 4, 2, 5))
    unfolded = tf.reshape(unfolded, (-1, new_h, new_w, transformer_dim))
    unfolded = unfolded[:, :height, :width, :]
    fused = tf.keras.layers.Conv2D(in_channels, 1, padding="same")(unfolded)
    fused = tf.keras.layers.Concatenate()([x, fused])
    fused = conv_bn_swish(fused, in_channels, kernel_size=3)
    return fused


def build_mobilevit_encoder(inputs, dropout_rate, encoder_dropout):
    x = conv_bn_swish(inputs, 16, kernel_size=3, strides=2)
    x = inverted_residual(x, 32, strides=1, expansion=4)
    skips = []
    skips.append(x)
    x = inverted_residual(x, 64, strides=2, expansion=4)
    x = inverted_residual(x, 64, strides=1, expansion=4)
    x = inverted_residual(x, 64, strides=1, expansion=4)
    if encoder_dropout:
        x = make_dropout(dropout_rate, True)(x)
    skips.append(x)
    x = inverted_residual(x, 96, strides=2, expansion=4)
    x = mobilevit_block(x, transformer_dim=144, num_heads=4, mlp_dim=288, depth=2, patch_size=2)
    if encoder_dropout:
        x = make_dropout(dropout_rate, True)(x)
    skips.append(x)
    x = inverted_residual(x, 128, strides=2, expansion=4)
    x = mobilevit_block(x, transformer_dim=192, num_heads=4, mlp_dim=384, depth=4, patch_size=2)
    if encoder_dropout:
        x = make_dropout(dropout_rate, True)(x)
    skips.append(x)
    x = inverted_residual(x, 160, strides=2, expansion=4)
    x = mobilevit_block(x, transformer_dim=240, num_heads=4, mlp_dim=480, depth=3, patch_size=2)
    if encoder_dropout:
        x = make_dropout(dropout_rate, True)(x)
    bottleneck = x
    skip_list = list(reversed(skips))
    return bottleneck, skip_list
