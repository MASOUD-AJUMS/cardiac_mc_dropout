import numpy as np
from scipy.optimize import minimize_scalar

from config import ECE_BINS, FOREGROUND_INDICES


def foreground_mask(true_labels):
    mask = np.zeros(true_labels.shape, dtype=bool)
    for index in FOREGROUND_INDICES:
        mask |= true_labels == index
    return mask


def gather_foreground(probs, true_labels):
    mask = foreground_mask(true_labels)
    selected_probs = probs[mask]
    selected_labels = true_labels[mask]
    confidences = selected_probs.max(axis=-1)
    predictions = selected_probs.argmax(axis=-1)
    correct = (predictions == selected_labels).astype(np.float64)
    return confidences, correct, selected_probs, selected_labels


def expected_calibration_error(confidences, correct, n_bins=ECE_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(confidences)
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_conf = confidences[in_bin].mean()
        avg_acc = correct[in_bin].mean()
        ece += (count / total) * abs(avg_conf - avg_acc)
    return ece


def adaptive_calibration_error(confidences, correct, n_bins=ECE_BINS):
    order = np.argsort(confidences)
    conf_sorted = confidences[order]
    correct_sorted = correct[order]
    splits = np.array_split(np.arange(len(conf_sorted)), n_bins)
    aece = 0.0
    total = len(conf_sorted)
    for split in splits:
        if len(split) == 0:
            continue
        avg_conf = conf_sorted[split].mean()
        avg_acc = correct_sorted[split].mean()
        aece += (len(split) / total) * abs(avg_conf - avg_acc)
    return aece


def brier_score(selected_probs, selected_labels, num_classes):
    one_hot = np.eye(num_classes)[selected_labels]
    return np.mean(np.sum((selected_probs - one_hot) ** 2, axis=-1))


def negative_log_likelihood(selected_probs, selected_labels, eps=1e-12):
    chosen = selected_probs[np.arange(len(selected_labels)), selected_labels]
    return -np.mean(np.log(np.clip(chosen, eps, 1.0)))


def calibration_report(probs, true_labels, num_classes, n_bins=ECE_BINS):
    confidences, correct, selected_probs, selected_labels = gather_foreground(probs, true_labels)
    return {
        "ece": expected_calibration_error(confidences, correct, n_bins),
        "aece": adaptive_calibration_error(confidences, correct, n_bins),
        "brier": brier_score(selected_probs, selected_labels, num_classes),
        "nll": negative_log_likelihood(selected_probs, selected_labels),
    }


def reliability_curve(confidences, correct, n_bins=ECE_BINS):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, accuracies, confidence_means, counts = [], [], [], []
    for lower, upper in zip(bins[:-1], bins[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        count = in_bin.sum()
        centers.append((lower + upper) / 2.0)
        counts.append(count)
        if count == 0:
            accuracies.append(np.nan)
            confidence_means.append(np.nan)
        else:
            accuracies.append(correct[in_bin].mean())
            confidence_means.append(confidences[in_bin].mean())
    return np.array(centers), np.array(accuracies), np.array(confidence_means), np.array(counts)


def fit_temperature(logits, labels, eps=1e-12):
    foreground = foreground_mask(labels)
    flat_logits = logits[foreground]
    flat_labels = labels[foreground]

    def nll(temperature):
        scaled = flat_logits / temperature
        scaled = scaled - scaled.max(axis=-1, keepdims=True)
        exp = np.exp(scaled)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        chosen = probs[np.arange(len(flat_labels)), flat_labels]
        return -np.mean(np.log(np.clip(chosen, eps, 1.0)))

    result = minimize_scalar(nll, bounds=(0.5, 5.0), method="bounded")
    return result.x


def apply_temperature(logits, temperature):
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=-1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=-1, keepdims=True)
