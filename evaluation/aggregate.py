import numpy as np

from config import FOREGROUND_INDICES, STRUCTURE_NAMES, NUM_CLASSES, MC_PASSES
from data.acdc import extract_slices
from evaluation.metrics import per_structure_metrics
from evaluation.mc_inference import mc_predict, deterministic_predict, predict_labels, case_uncertainty
from evaluation.calibration import calibration_report, foreground_mask


def slices_for_patient(patient, phase):
    pairs = extract_slices(patient, phase)
    images = np.stack([pair[0] for pair in pairs], axis=0).astype(np.float32)
    labels = np.stack([pair[1] for pair in pairs], axis=0).astype(np.int64)
    return images, labels


def predict_patient(model, patient, phase, use_mc, passes=MC_PASSES):
    images, labels = slices_for_patient(patient, phase)
    if use_mc:
        mean_probs, uncertainty = mc_predict(model, images, passes=passes)
    else:
        mean_probs = deterministic_predict(model, images)
        uncertainty = np.zeros(mean_probs.shape[:-1], dtype=np.float64)
    pred_labels = predict_labels(mean_probs)
    return images, labels, mean_probs, pred_labels, uncertainty


def evaluate_patient(model, patient, phase, use_mc, spacing=(1.0, 1.0)):
    images, labels, mean_probs, pred_labels, uncertainty = predict_patient(
        model, patient, phase, use_mc
    )
    structure = per_structure_metrics(pred_labels, labels, spacing)
    fg = foreground_mask(labels)
    unc = case_uncertainty(uncertainty, fg)
    return {
        "id": patient["id"],
        "group": patient["group"],
        "phase": phase,
        "structures": structure,
        "uncertainty": unc,
        "probs": mean_probs,
        "labels": labels,
        "pred_labels": pred_labels,
        "uncertainty_map": uncertainty,
    }


def evaluate_dataset(model, patients, use_mc, phases=("ed", "es")):
    records = []
    pooled_probs = []
    pooled_labels = []
    for patient in patients:
        for phase in phases:
            record = evaluate_patient(model, patient, phase, use_mc)
            pooled_probs.append(record["probs"].reshape(-1, NUM_CLASSES))
            pooled_labels.append(record["labels"].reshape(-1))
            records.append(record)
    probs = np.concatenate(pooled_probs, axis=0)
    labels = np.concatenate(pooled_labels, axis=0)
    calibration = calibration_report(probs, labels, NUM_CLASSES)
    return records, calibration


def mean_dsc_per_structure(records):
    sums = {name: [] for name in STRUCTURE_NAMES.values()}
    for record in records:
        for name, values in record["structures"].items():
            sums[name].append(values["dsc"])
    return {name: float(np.mean(values)) for name, values in sums.items()}


def overall_mean_dsc(records):
    values = []
    for record in records:
        for name in STRUCTURE_NAMES.values():
            values.append(record["structures"][name]["dsc"])
    return float(np.mean(values))
