
import os
import glob
import numpy as np
import nibabel as nib

from config import (
    CROP_SIZE,
    NUM_CLASSES,
    PATHOLOGIES,
    N_PER_CLASS_TRAIN,
    N_PER_CLASS_TEST,
    N_FOLDS,
    SEED,
    DATA_ROOT,
)


def read_info_cfg(patient_dir):
    info = {}
    cfg_path = os.path.join(patient_dir, "Info.cfg")
    with open(cfg_path, "r") as handle:
        for line in handle:
            if ":" in line:
                key, value = line.split(":", 1)
                info[key.strip()] = value.strip()
    return info


def list_patients(root):
    patients = []
    for split in ["training", "testing"]:
        split_dir = os.path.join(root, split)
        if not os.path.isdir(split_dir):
            continue
        for patient_dir in sorted(glob.glob(os.path.join(split_dir, "patient*"))):
            info = read_info_cfg(patient_dir)
            patients.append(
                {
                    "id": os.path.basename(patient_dir),
                    "dir": patient_dir,
                    "group": info.get("Group", "NOR"),
                    "ed": int(info.get("ED", 1)),
                    "es": int(info.get("ES", 1)),
                }
            )
    return patients


def stratified_partition(patients, n_train_per_class, n_test_per_class, seed):
    rng = np.random.default_rng(seed)
    by_group = {p: [] for p in PATHOLOGIES}
    for patient in patients:
        group = patient["group"]
        if group in by_group:
            by_group[group].append(patient)
    train, test = [], []
    for group in PATHOLOGIES:
        members = by_group[group]
        order = rng.permutation(len(members))
        selected = [members[i] for i in order]
        train.extend(selected[:n_train_per_class])
        test.extend(selected[n_train_per_class:n_train_per_class + n_test_per_class])
    return train, test


def stratified_folds(train_patients, n_folds, seed):
    rng = np.random.default_rng(seed)
    by_group = {p: [] for p in PATHOLOGIES}
    for patient in train_patients:
        by_group[patient["group"]].append(patient)
    folds = [[] for _ in range(n_folds)]
    for group in PATHOLOGIES:
        members = by_group[group]
        order = rng.permutation(len(members))
        for position, index in enumerate(order):
            folds[position % n_folds].append(members[index])
    return folds


def load_phase_volume(patient, phase):
    frame = patient["ed"] if phase == "ed" else patient["es"]
    base = os.path.join(patient["dir"], f"{patient['id']}_frame{frame:02d}")
    image = nib.load(base + ".nii.gz").get_fdata().astype(np.float32)
    label = nib.load(base + "_gt.nii.gz").get_fdata().astype(np.int64)
    return image, label


def center_crop(array, size):
    target_h, target_w = size
    h, w = array.shape[:2]
    top = max((h - target_h) // 2, 0)
    left = max((w - target_w) // 2, 0)
    cropped = array[top:top + target_h, left:left + target_w]
    pad_h = target_h - cropped.shape[0]
    pad_w = target_w - cropped.shape[1]
    if pad_h > 0 or pad_w > 0:
        pad_config = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (cropped.ndim - 2)
        cropped = np.pad(cropped, pad_config, mode="constant")
    return cropped


def zscore(slice_array):
    mean = slice_array.mean()
    std = slice_array.std()
    if std < 1e-6:
        return slice_array - mean
    return (slice_array - mean) / std


def extract_slices(patient, phase):
    image, label = load_phase_volume(patient, phase)
    slices = []
    for index in range(image.shape[2]):
        img_slice = center_crop(image[:, :, index], CROP_SIZE)
        lbl_slice = center_crop(label[:, :, index], CROP_SIZE)
        img_slice = zscore(img_slice)
        slices.append((img_slice[..., None], lbl_slice))
    return slices


def build_slice_dataset(patients, phases=("ed", "es")):
    images, labels, meta = [], [], []
    for patient in patients:
        for phase in phases:
            for img_slice, lbl_slice in extract_slices(patient, phase):
                images.append(img_slice)
                labels.append(lbl_slice)
                meta.append({"id": patient["id"], "group": patient["group"], "phase": phase})
    images = np.stack(images, axis=0).astype(np.float32)
    labels = np.stack(labels, axis=0).astype(np.int64)
    return images, labels, meta


def one_hot(labels, num_classes=NUM_CLASSES):
    return np.eye(num_classes, dtype=np.float32)[labels]


def load_partitions(root=DATA_ROOT, seed=SEED):
    patients = list_patients(root)
    train, test = stratified_partition(patients, N_PER_CLASS_TRAIN, N_PER_CLASS_TEST, seed)
    folds = stratified_folds(train, N_FOLDS, seed)
    return {"train": train, "test": test, "folds": folds}
