import os
import numpy as np

from config import (
    OUTPUT_ROOT,
    PRIMARY_BACKBONE,
    SEED,
    MC_PASSES,
)
from data.acdc import load_partitions, build_slice_dataset
from training.train import train_model
from evaluation.aggregate import evaluate_dataset, overall_mean_dsc
from evaluation.mc_inference import mc_predict, predict_labels
from evaluation.aggregate import slices_for_patient
from evaluation.metrics import per_structure_metrics
from evaluation.calibration import calibration_report, foreground_mask
from config import NUM_CLASSES

DROPOUT_RATES = [0.1, 0.2, 0.3, 0.4, 0.5]
PASS_COUNTS = [5, 10, 20, 30, 50]
PLACEMENTS = ["encoder", "decoder", "both"]


def validation_set(seed=SEED):
    partitions = load_partitions(seed=seed)
    folds = partitions["folds"]
    val_patients = folds[0]
    train_patients = []
    for index, fold in enumerate(folds):
        if index != 0:
            train_patients.extend(fold)
    return train_patients, val_patients


def dropout_rate_ablation(backbone=PRIMARY_BACKBONE, seed=SEED):
    train_patients, val_patients = validation_set(seed)
    train_images, train_labels, _ = build_slice_dataset(train_patients)
    val_images, val_labels, _ = build_slice_dataset(val_patients)
    results = []
    for rate in DROPOUT_RATES:
        tag = f"ablation_rate_{backbone}_{rate}_seed{seed}"
        checkpoint = os.path.join(OUTPUT_ROOT, "checkpoints", tag + ".weights.h5")
        model, _ = train_model(
            backbone, train_images, train_labels, val_images, val_labels,
            checkpoint, dropout_active=True, dropout_rate=rate,
            placement="encoder", seed=seed
        )
        records, calibration = evaluate_dataset(model, val_patients, use_mc=True)
        results.append({"rate": rate, "dsc": overall_mean_dsc(records), "ece": calibration["ece"]})
    return results


def pass_count_ablation(backbone=PRIMARY_BACKBONE, seed=SEED):
    train_patients, val_patients = validation_set(seed)
    train_images, train_labels, _ = build_slice_dataset(train_patients)
    val_images, val_labels, _ = build_slice_dataset(val_patients)
    tag = f"ablation_passes_{backbone}_seed{seed}"
    checkpoint = os.path.join(OUTPUT_ROOT, "checkpoints", tag + ".weights.h5")
    model, _ = train_model(
        backbone, train_images, train_labels, val_images, val_labels,
        checkpoint, dropout_active=True, placement="encoder", seed=seed
    )
    results = []
    for passes in PASS_COUNTS:
        pooled_probs, pooled_labels, dscs = [], [], []
        for patient in val_patients:
            for phase in ("ed", "es"):
                images, labels = slices_for_patient(patient, phase)
                mean_probs, _ = mc_predict(model, images, passes=passes)
                pred_labels = predict_labels(mean_probs)
                structure = per_structure_metrics(pred_labels, labels)
                for values in structure.values():
                    dscs.append(values["dsc"])
                pooled_probs.append(mean_probs.reshape(-1, NUM_CLASSES))
                pooled_labels.append(labels.reshape(-1))
        probs = np.concatenate(pooled_probs, axis=0)
        labels_all = np.concatenate(pooled_labels, axis=0)
        calibration = calibration_report(probs, labels_all, NUM_CLASSES)
        results.append({"passes": passes, "dsc": float(np.mean(dscs)), "ece": calibration["ece"]})
    return results


def placement_ablation(backbones=None, seed=SEED):
    if backbones is None:
        from config import BACKBONES
        backbones = BACKBONES
    train_patients, val_patients = validation_set(seed)
    train_images, train_labels, _ = build_slice_dataset(train_patients)
    val_images, val_labels, _ = build_slice_dataset(val_patients)
    results = []
    for backbone in backbones:
        for placement in PLACEMENTS:
            tag = f"ablation_place_{backbone}_{placement}_seed{seed}"
            checkpoint = os.path.join(OUTPUT_ROOT, "checkpoints", tag + ".weights.h5")
            model, _ = train_model(
                backbone, train_images, train_labels, val_images, val_labels,
                checkpoint, dropout_active=True, placement=placement, seed=seed
            )
            records, calibration = evaluate_dataset(model, val_patients, use_mc=True)
            results.append(
                {
                    "backbone": backbone,
                    "placement": placement,
                    "dsc": overall_mean_dsc(records),
                    "ece": calibration["ece"],
                }
            )
    return results
