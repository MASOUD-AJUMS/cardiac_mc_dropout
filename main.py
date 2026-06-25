import os
import json
import argparse

import numpy as np

from config import (
    BACKBONES,
    PRIMARY_BACKBONE,
    OUTPUT_ROOT,
    SEED,
    NUM_CLASSES,
)
from data.acdc import load_partitions
from experiments.cross_validation import run_cross_validation, retrain_on_full_training_set
from experiments.ablation import (
    dropout_rate_ablation,
    pass_count_ablation,
    placement_ablation,
)
from experiments.per_pathology import per_pathology_dsc, per_pathology_calibration
from experiments.robustness import multi_seed_delta_ece
from evaluation.aggregate import evaluate_dataset, mean_dsc_per_structure, overall_mean_dsc
from evaluation.calibration import calibration_report, fit_temperature, apply_temperature
from evaluation.bin_sweep import max_ece_spread
from evaluation.statistics import per_structure_significance
from evaluation.quality_control import qc_enrichment, case_is_failure
from models import build_logit_model


def save_json(obj, name):
    os.makedirs(os.path.join(OUTPUT_ROOT, "results"), exist_ok=True)
    path = os.path.join(OUTPUT_ROOT, "results", name)
    with open(path, "w") as handle:
        json.dump(obj, handle, indent=2, default=float)
    return path


def run_cross_validation_stage():
    summary = {}
    for backbone in BACKBONES:
        summary[backbone] = {
            "baseline": run_cross_validation(backbone, dropout_active=False),
            "mc": run_cross_validation(backbone, dropout_active=True),
        }
    save_json(summary, "cross_validation.json")
    return summary


def evaluate_test_stage():
    partitions = load_partitions(seed=SEED)
    test_patients = partitions["test"]
    results = {}
    for backbone in BACKBONES:
        base_model, _ = retrain_on_full_training_set(backbone, dropout_active=False)
        base_records, base_cal = evaluate_dataset(base_model, test_patients, use_mc=False)
        mc_model, _ = retrain_on_full_training_set(backbone, dropout_active=True)
        mc_records, mc_cal = evaluate_dataset(mc_model, test_patients, use_mc=True)
        logit_model = build_logit_model(base_model)
        temp_result = temperature_scaling_stage(logit_model, partitions, test_patients)
        results[backbone] = {
            "baseline_dsc": mean_dsc_per_structure(base_records),
            "mc_dsc": mean_dsc_per_structure(mc_records),
            "baseline_calibration": base_cal,
            "mc_calibration": mc_cal,
            "temperature": temp_result,
            "overall_mc_dsc": overall_mean_dsc(mc_records),
        }
        if backbone == PRIMARY_BACKBONE:
            results[backbone]["significance"] = collect_significance(base_records, mc_records)
            results[backbone]["pathology_dsc"] = per_pathology_dsc(mc_records)
            rows, disparity = per_pathology_calibration(base_records, mc_records)
            results[backbone]["pathology_calibration"] = rows
            results[backbone]["calibration_disparity"] = disparity
            results[backbone]["qc"] = quality_control_stage(mc_records)
            mc_probs, mc_labels = pooled(mc_records)
            spread, sweep = max_ece_spread(mc_probs, mc_labels)
            results[backbone]["bin_sweep"] = {"spread": spread, "values": sweep}
    save_json(results, "test_results.json")
    return results


def pooled(records):
    probs = np.concatenate([r["probs"].reshape(-1, NUM_CLASSES) for r in records], axis=0)
    labels = np.concatenate([r["labels"].reshape(-1) for r in records], axis=0)
    return probs, labels


def temperature_scaling_stage(logit_model, partitions, test_patients):
    val_patients = partitions["folds"][0]
    val_logits, val_labels = collect_logits(logit_model, val_patients)
    temperature = fit_temperature(val_logits, val_labels)
    test_logits, test_labels = collect_logits(logit_model, test_patients)
    scaled = apply_temperature(test_logits, temperature)
    calibration = calibration_report(
        scaled.reshape(-1, NUM_CLASSES), test_labels.reshape(-1), NUM_CLASSES
    )
    return {"temperature": float(temperature), "calibration": calibration}


def collect_logits(logit_model, patients):
    from evaluation.aggregate import slices_for_patient
    logits_list, labels_list = [], []
    for patient in patients:
        for phase in ("ed", "es"):
            images, labels = slices_for_patient(patient, phase)
            logits = logit_model.predict(images, verbose=0)
            logits_list.append(logits)
            labels_list.append(labels)
    return np.concatenate(logits_list, axis=0), np.concatenate(labels_list, axis=0)


def collect_significance(base_records, mc_records):
    output = {}
    for structure in ("LV", "RV", "Myo"):
        for phase in ("ed", "es"):
            key = f"{structure}_{phase}"
            output[key] = per_structure_significance(base_records, mc_records, structure, phase)
    return output


def quality_control_stage(mc_records):
    by_patient = {}
    for record in mc_records:
        pid = record["id"]
        by_patient.setdefault(pid, {"uncertainty": [], "dscs": []})
        by_patient[pid]["uncertainty"].append(record["uncertainty"])
        for values in record["structures"].values():
            by_patient[pid]["dscs"].append(values["dsc"])
    uncertainties, failures = [], []
    for pid, data in by_patient.items():
        uncertainties.append(float(np.mean(data["uncertainty"])))
        failures.append(case_is_failure(data["dscs"]))
    return qc_enrichment(uncertainties, failures)


def run_ablation_stage():
    output = {
        "dropout_rate": dropout_rate_ablation(),
        "pass_count": pass_count_ablation(),
        "placement": placement_ablation(),
    }
    save_json(output, "ablation.json")
    return output


def run_robustness_stage():
    output = multi_seed_delta_ece()
    save_json(output, "robustness.json")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "cv", "test", "ablation", "robustness"])
    args = parser.parse_args()
    if args.stage in ("all", "cv"):
        run_cross_validation_stage()
    if args.stage in ("all", "test"):
        evaluate_test_stage()
    if args.stage in ("all", "ablation"):
        run_ablation_stage()
    if args.stage in ("all", "robustness"):
        run_robustness_stage()


if __name__ == "__main__":
    main()
