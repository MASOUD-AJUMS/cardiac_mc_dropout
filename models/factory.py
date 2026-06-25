import tensorflow as tf

from config import CROP_SIZE, NUM_CLASSES, DROPOUT_RATE, DROPOUT_PLACEMENT
from models.encoders import build_efficientnet_encoder
from models.swin import build_swin_encoder
from models.mobilevit import build_mobilevit_encoder

DECODER_FILTERS = [256, 128, 64, 32, 16]


def resolve_placement(placement):
    encoder_dropout = placement in ("encoder", "both")
    decoder_dropout = placement in ("decoder", "both")
    return encoder_dropout, decoder_dropout


def build_model(backbone, dropout_active=True, dropout_rate=DROPOUT_RATE,
                placement=DROPOUT_PLACEMENT, num_classes=NUM_CLASSES,
                input_size=CROP_SIZE):
    from models.linknet import build_decoder

    if dropout_active:
        encoder_dropout, decoder_dropout = resolve_placement(placement)
    else:
        encoder_dropout, decoder_dropout = False, False

    inputs = tf.keras.Input(shape=(input_size[0], input_size[1], 1))

    if backbone in ("efficientnet-b3", "efficientnet-b4"):
        bottleneck, skips = build_efficientnet_encoder(
            inputs, backbone, dropout_rate, encoder_dropout
        )
    elif backbone == "swinunet":
        bottleneck, skips = build_swin_encoder(
            inputs, embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
            window_size=7, dropout_rate=dropout_rate, encoder_dropout=encoder_dropout
        )
    elif backbone == "mobilevit-s":
        bottleneck, skips = build_mobilevit_encoder(
            inputs, dropout_rate, encoder_dropout
        )
    else:
        raise ValueError(backbone)

    logits = build_decoder(
        bottleneck, skips, num_classes, DECODER_FILTERS,
        dropout_rate, decoder_dropout, target_size=input_size
    )
    outputs = tf.keras.layers.Softmax()(logits)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{backbone}_{placement}")


def build_logit_model(model):
    softmax_layer = model.layers[-1]
    logits = softmax_layer.input
    return tf.keras.Model(inputs=model.input, outputs=logits)
