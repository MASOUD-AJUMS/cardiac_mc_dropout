import numpy as np
from scipy.stats import pearsonr

from config import LV_INDEX, RV_INDEX


def chamber_volume(label_volume, index, voxel_volume_ml):
    return float(np.sum(label_volume == index) * voxel_volume_ml)


def ejection_fraction(edv, esv):
    if edv <= 0:
        return 0.0
    return 100.0 * (edv - esv) / edv


def patient_indices(ed_labels, es_labels, voxel_volume_ml):
    lv_edv = chamber_volume(ed_labels, LV_INDEX, voxel_volume_ml)
    lv_esv = chamber_volume(es_labels, LV_INDEX, voxel_volume_ml)
    rv_edv = chamber_volume(ed_labels, RV_INDEX, voxel_volume_ml)
    rv_esv = chamber_volume(es_labels, RV_INDEX, voxel_volume_ml)
    return {
        "lv_edv": lv_edv,
        "lv_esv": lv_esv,
        "lv_ef": ejection_fraction(lv_edv, lv_esv),
        "rv_edv": rv_edv,
        "rv_esv": rv_esv,
        "rv_ef": ejection_fraction(rv_edv, rv_esv),
    }


def agreement_stats(predicted, reference):
    predicted = np.asarray(predicted)
    reference = np.asarray(reference)
    r, _ = pearsonr(predicted, reference)
    mae = np.mean(np.abs(predicted - reference))
    diff = predicted - reference
    bias = np.mean(diff)
    loa = 1.96 * np.std(diff)
    return {"pearson_r": r, "mae": mae, "bias": bias, "loa": loa}
