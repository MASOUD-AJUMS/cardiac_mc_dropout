import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

from config import FOREGROUND_INDICES, STRUCTURE_NAMES


def dice_score(pred_mask, true_mask):
    intersection = np.sum(pred_mask & true_mask)
    total = np.sum(pred_mask) + np.sum(true_mask)
    if total == 0:
        return 1.0
    return 2.0 * intersection / total


def surface_distances(pred_mask, true_mask, spacing):
    if pred_mask.sum() == 0 or true_mask.sum() == 0:
        return None
    pred_border = pred_mask ^ binary_erode(pred_mask)
    true_border = true_mask ^ binary_erode(true_mask)
    dt_true = distance_transform_edt(~true_border, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    pred_to_true = dt_true[pred_border]
    true_to_pred = dt_pred[true_border]
    return np.concatenate([pred_to_true, true_to_pred])


def binary_erode(mask):
    from scipy.ndimage import binary_erosion
    return binary_erosion(mask, border_value=0)


def hd95(pred_mask, true_mask, spacing):
    distances = surface_distances(pred_mask, true_mask, spacing)
    if distances is None or distances.size == 0:
        return np.nan
    return np.percentile(distances, 95)


def precision_recall(pred_mask, true_mask):
    tp = np.sum(pred_mask & true_mask)
    fp = np.sum(pred_mask & ~true_mask)
    fn = np.sum(~pred_mask & true_mask)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall


def per_structure_metrics(pred_labels, true_labels, spacing=(1.0, 1.0)):
    results = {}
    for index in FOREGROUND_INDICES:
        name = STRUCTURE_NAMES[index]
        pred_mask = pred_labels == index
        true_mask = true_labels == index
        precision, recall = precision_recall(pred_mask, true_mask)
        results[name] = {
            "dsc": dice_score(pred_mask, true_mask),
            "hd95": hd95(pred_mask, true_mask, spacing),
            "precision": precision,
            "recall": recall,
        }
    return results
