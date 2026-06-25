import numpy as np

from config import BIN_SWEEP, NUM_CLASSES
from evaluation.calibration import gather_foreground, expected_calibration_error


def ece_bin_sweep(probs, true_labels, bins=BIN_SWEEP):
    confidences, correct, _, _ = gather_foreground(probs, true_labels)
    return {n: expected_calibration_error(confidences, correct, n) for n in bins}


def max_ece_spread(probs, true_labels, bins=BIN_SWEEP):
    sweep = ece_bin_sweep(probs, true_labels, bins)
    values = list(sweep.values())
    return max(values) - min(values), sweep
