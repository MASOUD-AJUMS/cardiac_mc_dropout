import os
import numpy as np

from config import (
    N_FOLDS,
    OUTPUT_ROOT,
    BACKBONES,
    DROPOUT_RATE,
    DROPOUT_PLACEMENT,
    SEED,
)
from data.acdc import load_partitions, build_slice_dataset
from training.train import train_model
from evaluation.aggregate import evaluate_dataset, mean_dsc_per_structure


def fold_split(folds, held_out):
    val_patients = folds[held_out]
    train_patients = []
    for index, fold in enumerate(folds):
        if index != held_out:
            train_patients.extend(fold)
    return train_patients, val_patients


def run_cross_validation(backbone, dropout_active, placement=DROPOUT_PLACEMENT,
                         dropout_rate=DROPOUT_RATE, seed=SEED):
    partitions = load_partitions(seed=seed)
    folds = partitions["folds"]
    fold_results = []
    for held_out in range(N_FOLDS):
        train_patients, val_patients = fold_split(folds, held_out)
        train_images, train_labels, _ = build_slice_dataset(train_patients)
        val_images, val_labels, _ = build_slice_dataset(val_patients)
        tag = f"{backbone}_{'mc' if dropout_active else 'base'}_{placement}_seed{seed}_fold{held_out}"
        checkpoint = os.path.join(OUTPUT_ROOT, "checkpoints", tag + ".weights.h5")
        model, history = train_model(
            backbone, train_images, train_labels, val_images, val_labels,
            checkpoint, dropout_active=dropout_active,
            dropout_rate=dropout_rate, placement=placement, seed=seed
        )
        records, calibration = evaluate_dataset(model, val_patients, use_mc=dropout_active)
        fold_results.append(
            {
                "fold": held_out,
                "dsc": mean_dsc_per_structure(records),
                "calibration": calibration,
                "history": history,
            }
        )
    return fold_results


def retrain_on_full_training_set(backbone, dropout_active, placement=DROPOUT_PLACEMENT,
                                 dropout_rate=DROPOUT_RATE, seed=SEED):
    partitions = load_partitions(seed=seed)
    train_patients = partitions["train"]
    folds = partitions["folds"]
    held_out = folds[0]
    inner_train = [p for p in train_patients if p not in held_out]
    train_images, train_labels, _ = build_slice_dataset(inner_train)
    val_images, val_labels, _ = build_slice_dataset(held_out)
    tag = f"{backbone}_{'mc' if dropout_active else 'base'}_{placement}_seed{seed}_full"
    checkpoint = os.path.join(OUTPUT_ROOT, "checkpoints", tag + ".weights.h5")
    model, history = train_model(
        backbone, train_images, train_labels, val_images, val_labels,
        checkpoint, dropout_active=dropout_active,
        dropout_rate=dropout_rate, placement=placement, seed=seed
    )
    return model, checkpoint
