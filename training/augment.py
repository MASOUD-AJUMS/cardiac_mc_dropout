import numpy as np
import scipy.ndimage as ndi

from config import (
    AUG_FLIP_PROB,
    AUG_ROTATION_DEG,
    AUG_SCALE_RANGE,
    AUG_BRIGHTNESS,
    AUG_CONTRAST,
)


def augment_pair(image, label, rng):
    if rng.random() < AUG_FLIP_PROB:
        image = image[:, ::-1, :]
        label = label[:, ::-1]
    if rng.random() < AUG_FLIP_PROB:
        image = image[::-1, :, :]
        label = label[::-1, :]
    angle = rng.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG)
    scale = rng.uniform(*AUG_SCALE_RANGE)
    image = apply_affine(image, angle, scale, order=1)
    label = apply_affine_label(label, angle, scale)
    brightness = rng.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
    contrast = 1.0 + rng.uniform(-AUG_CONTRAST, AUG_CONTRAST)
    image = image * contrast + brightness
    return image.astype(np.float32), label.astype(np.int64)


def apply_affine(image, angle, scale, order):
    rotated = ndi.rotate(image[..., 0], angle, reshape=False, order=order, mode="constant")
    zoomed = zoom_to_shape(rotated, scale, order)
    return zoomed[..., None]


def apply_affine_label(label, angle, scale):
    rotated = ndi.rotate(label, angle, reshape=False, order=0, mode="constant")
    zoomed = zoom_to_shape(rotated, scale, order=0)
    return zoomed


def zoom_to_shape(array, scale, order):
    original = array.shape
    zoomed = ndi.zoom(array, scale, order=order, mode="constant")
    output = np.zeros(original, dtype=array.dtype)
    src_h = min(zoomed.shape[0], original[0])
    src_w = min(zoomed.shape[1], original[1])
    sy = (zoomed.shape[0] - src_h) // 2
    sx = (zoomed.shape[1] - src_w) // 2
    dy = (original[0] - src_h) // 2
    dx = (original[1] - src_w) // 2
    output[dy:dy + src_h, dx:dx + src_w] = zoomed[sy:sy + src_h, sx:sx + src_w]
    return output


def augment_batch(images, labels, seed):
    rng = np.random.default_rng(seed)
    aug_images = np.empty_like(images)
    aug_labels = np.empty_like(labels)
    for index in range(images.shape[0]):
        aug_images[index], aug_labels[index] = augment_pair(images[index], labels[index], rng)
    return aug_images, aug_labels
