import tensorflow as tf

from models.dropout import make_dropout

EFFICIENTNET_SKIPS = [
    "block6a_expand_activation",
    "block4a_expand_activation",
    "block3a_expand_activation",
    "block2a_expand_activation",
]
EFFICIENTNET_OUTPUT = "top_activation"


def build_efficientnet_encoder(inputs, variant, dropout_rate, encoder_dropout):
    rgb = tf.keras.layers.Conv2D(3, 1, padding="same")(inputs)
    if variant == "efficientnet-b3":
        backbone = tf.keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet", input_tensor=rgb
        )
    elif variant == "efficientnet-b4":
        backbone = tf.keras.applications.EfficientNetB4(
            include_top=False, weights="imagenet", input_tensor=rgb
        )
    else:
        raise ValueError(variant)
    bottleneck = backbone.get_layer(EFFICIENTNET_OUTPUT).output
    if encoder_dropout:
        bottleneck = make_dropout(dropout_rate, True)(bottleneck)
    skips = []
    for name in EFFICIENTNET_SKIPS:
        feature = backbone.get_layer(name).output
        if encoder_dropout:
            feature = make_dropout(dropout_rate, True)(feature)
        skips.append(feature)
    return bottleneck, skips
