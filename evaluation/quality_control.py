import numpy as np

from config import QC_THRESHOLDS, FAILURE_DSC


def wilson_interval(successes, total, z=1.96):
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    denom = 1 + z ** 2 / total
    center = (phat + z ** 2 / (2 * total)) / denom
    margin = z * np.sqrt(phat * (1 - phat) / total + z ** 2 / (4 * total ** 2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def case_is_failure(structure_dscs, threshold=FAILURE_DSC):
    return any(value < threshold for value in structure_dscs)


def qc_enrichment(case_uncertainties, failures, thresholds=QC_THRESHOLDS):
    n = len(case_uncertainties)
    order = np.argsort(case_uncertainties)[::-1]
    failures = np.asarray(failures, dtype=bool)
    total_failures = failures.sum()
    base_rate = total_failures / n if n > 0 else 0.0
    rows = []
    for fraction in thresholds:
        k = max(int(round(fraction * n)), 1)
        flagged = order[:k]
        not_flagged = order[k:]
        captured = failures[flagged].sum()
        flagged_failure_rate = captured / k if k > 0 else 0.0
        enrichment = flagged_failure_rate / base_rate if base_rate > 0 else 0.0
        tn = (~failures[not_flagged]).sum()
        fn = failures[not_flagged].sum()
        tp = captured
        fp = k - captured
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rows.append(
            {
                "fraction": fraction,
                "flagged": k,
                "captured": int(captured),
                "total_failures": int(total_failures),
                "enrichment": enrichment,
                "npv": npv,
                "npv_ci": wilson_interval(tn, tn + fn),
                "ppv": ppv,
                "ppv_ci": wilson_interval(tp, tp + fp),
            }
        )
    return rows


def error_detection_auroc(scores, errors):
    scores = np.asarray(scores, dtype=np.float64)
    errors = np.asarray(errors, dtype=bool)
    positives = errors.sum()
    negatives = (~errors).sum()
    if positives == 0 or negatives == 0:
        return np.nan
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    rank_sum_pos = ranks[errors].sum()
    auc = (rank_sum_pos - positives * (positives + 1) / 2) / (positives * negatives)
    return auc


def voxel_error_auroc(uncertainty, pred_labels, true_labels, foreground):
    scores = uncertainty[foreground].ravel()
    errors = (pred_labels[foreground] != true_labels[foreground]).ravel()
    return error_detection_auroc(scores, errors)
