import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def paired_tests(values_a, values_b):
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    t_stat, t_p = ttest_rel(values_a, values_b)
    try:
        w_stat, w_p = wilcoxon(values_a, values_b)
    except ValueError:
        w_stat, w_p = np.nan, 1.0
    return {"t_stat": t_stat, "t_p": t_p, "wilcoxon_stat": w_stat, "wilcoxon_p": w_p}


def bonferroni_threshold(alpha, n_comparisons):
    return alpha / n_comparisons


def per_structure_significance(baseline_records, mc_records, structure, phase, alpha=0.05):
    base = [r["structures"][structure]["dsc"] for r in baseline_records if r["phase"] == phase]
    mc = [r["structures"][structure]["dsc"] for r in mc_records if r["phase"] == phase]
    result = paired_tests(mc, base)
    threshold = bonferroni_threshold(alpha, 3)
    result["threshold"] = threshold
    result["significant"] = result["t_p"] < threshold
    return result
