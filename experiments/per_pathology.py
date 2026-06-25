import numpy as np

from config import (
    PATHOLOGIES,
    NUM_CLASSES,
    BOOTSTRAP_RESAMPLES,
    SEED,
)
from evaluation.calibration import calibration_report


def group_records(records):
    grouped = {p: [] for p in PATHOLOGIES}
    for record in records:
        grouped[record["group"]].append(record)
    return grouped


def pooled_calibration(records):
    probs = np.concatenate([r["probs"].reshape(-1, NUM_CLASSES) for r in records], axis=0)
    labels = np.concatenate([r["labels"].reshape(-1) for r in records], axis=0)
    return calibration_report(probs, labels, NUM_CLASSES)


def per_pathology_dsc(records):
    grouped = group_records(records)
    summary = {}
    for group, group_records_list in grouped.items():
        per_structure = {"LV": [], "RV": [], "Myo": []}
        for record in group_records_list:
            for name, values in record["structures"].items():
                per_structure[name].append(values["dsc"])
        summary[group] = {name: float(np.mean(v)) for name, v in per_structure.items()}
        summary[group]["mean"] = float(np.mean([np.mean(v) for v in per_structure.values()]))
        summary[group]["uncertainty"] = float(
            np.mean([r["uncertainty"] for r in group_records_list])
        )
    return summary


def patient_level_probs_labels(records):
    bundle = {}
    for record in records:
        key = record["id"]
        bundle.setdefault(key, {"probs": [], "labels": [], "group": record["group"]})
        bundle[key]["probs"].append(record["probs"].reshape(-1, NUM_CLASSES))
        bundle[key]["labels"].append(record["labels"].reshape(-1))
    for key in bundle:
        bundle[key]["probs"] = np.concatenate(bundle[key]["probs"], axis=0)
        bundle[key]["labels"] = np.concatenate(bundle[key]["labels"], axis=0)
    return bundle


def bootstrap_delta_ece(baseline_records, mc_records, group, seed=SEED,
                        resamples=BOOTSTRAP_RESAMPLES):
    base_bundle = patient_level_probs_labels(baseline_records)
    mc_bundle = patient_level_probs_labels(mc_records)
    patient_ids = [pid for pid, value in base_bundle.items() if value["group"] == group]
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(resamples):
        sample = rng.choice(patient_ids, size=len(patient_ids), replace=True)
        base_probs = np.concatenate([base_bundle[pid]["probs"] for pid in sample], axis=0)
        base_labels = np.concatenate([base_bundle[pid]["labels"] for pid in sample], axis=0)
        mc_probs = np.concatenate([mc_bundle[pid]["probs"] for pid in sample], axis=0)
        mc_labels = np.concatenate([mc_bundle[pid]["labels"] for pid in sample], axis=0)
        base_ece = calibration_report(base_probs, base_labels, NUM_CLASSES)["ece"]
        mc_ece = calibration_report(mc_probs, mc_labels, NUM_CLASSES)["ece"]
        deltas.append(base_ece - mc_ece)
    deltas = np.array(deltas)
    return float(deltas.mean()), (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))


def per_pathology_calibration(baseline_records, mc_records):
    base_grouped = group_records(baseline_records)
    mc_grouped = group_records(mc_records)
    rows = []
    mc_eces = []
    for group in PATHOLOGIES:
        base_cal = pooled_calibration(base_grouped[group])
        mc_cal = pooled_calibration(mc_grouped[group])
        delta_mean, delta_ci = bootstrap_delta_ece(baseline_records, mc_records, group)
        mc_eces.append(mc_cal["ece"])
        rows.append(
            {
                "group": group,
                "ece_baseline": base_cal["ece"],
                "ece_mc": mc_cal["ece"],
                "delta_ece": delta_mean,
                "delta_ci": delta_ci,
            }
        )
    disparity = max(mc_eces) - min(mc_eces)
    return rows, disparity
