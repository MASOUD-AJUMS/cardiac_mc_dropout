import numpy as np

from config import ROBUSTNESS_SEEDS, PRIMARY_BACKBONE, NUM_CLASSES
from experiments.cross_validation import retrain_on_full_training_set
from data.acdc import load_partitions
from evaluation.aggregate import evaluate_dataset


def multi_seed_delta_ece(backbone=PRIMARY_BACKBONE, seeds=ROBUSTNESS_SEEDS):
    deltas = []
    for seed in seeds:
        partitions = load_partitions(seed=seed)
        test_patients = partitions["test"]
        base_model, _ = retrain_on_full_training_set(backbone, dropout_active=False, seed=seed)
        _, base_cal = evaluate_dataset(base_model, test_patients, use_mc=False)
        mc_model, _ = retrain_on_full_training_set(backbone, dropout_active=True, seed=seed)
        _, mc_cal = evaluate_dataset(mc_model, test_patients, use_mc=True)
        deltas.append(base_cal["ece"] - mc_cal["ece"])
    deltas = np.array(deltas)
    return {
        "mean": float(deltas.mean()),
        "std": float(deltas.std(ddof=1)),
        "per_seed": deltas.tolist(),
    }


def bootstrap_ci(values, resamples=1000, seed=0):
    rng = np.random.default_rng(seed)
    values = np.asarray(values)
    means = []
    for _ in range(resamples):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
