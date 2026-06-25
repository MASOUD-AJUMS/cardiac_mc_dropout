import tensorflow as tf

from models.dropout import make_dropout


def conv_bn_relu(x, filters, kernel_size=3, strides=1):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def deconv_bn_relu(x, filters):
    x = tf.keras.layers.Conv2DTranspose(filters, 3, strides=2, padding="same", use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x


def match_and_add(x, skip):
    if skip.shape[1] != x.shape[1] or skip.shape[2] != x.shape[2]:
        skip = tf.keras.layers.Resizing(int(x.shape[1]), int(x.shape[2]), interpolation="bilinear")(skip)
    if skip.shape[-1] != x.shape[-1]:
        skip = tf.keras.layers.Conv2D(int(x.shape[-1]), 1, padding="same", use_bias=False)(skip)
    return tf.keras.layers.Add()([x, skip])


def linknet_block(x, skip, out_filters):
    inner = max(int(x.shape[-1]) // 4, 1)
    x = conv_bn_relu(x, inner, kernel_size=1)
    x = deconv_bn_relu(x, inner)
    x = conv_bn_relu(x, out_filters, kernel_size=1)
    if skip is not None:
        x = match_and_add(x, skip)
    return x


def build_decoder(bottleneck, skips, num_classes, decoder_filters, dropout_rate,
                  decoder_dropout, target_size):
    x = bottleneck
    skips = list(skips)
    for index, filters in enumerate(decoder_filters):
        skip = skips[index] if index < len(skips) else None
        x = linknet_block(x, skip, filters)
        if decoder_dropout:
            x = make_dropout(dropout_rate, True)(x)
    if x.shape[1] != target_size[0] or x.shape[2] != target_size[1]:
        x = tf.keras.layers.Resizing(target_size[0], target_size[1], interpolation="bilinear")(x)
    x = conv_bn_relu(x, decoder_filters[-1], kernel_size=3)
    if decoder_dropout:
        x = make_dropout(dropout_rate, True)(x)
    logits = tf.keras.layers.Conv2D(num_classes, 1, padding="same")(x)
    return logits
