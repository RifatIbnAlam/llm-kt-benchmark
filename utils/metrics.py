"""
Evaluation utilities — metrics with bootstrap confidence intervals.

Why bootstrap CIs matter for this paper:
Reviewers will ask whether the gap between DKT and the LLM zero-shot result
is significant. With ~91K test predictions, even tiny AUC differences are
"statistically significant" by classical tests, so we report 95% bootstrap
CIs over predictions and per-student bootstraps for the LLM-vs-KT comparison.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score


def bootstrap_auc_ci(labels, preds, n_boot=1000, ci=0.95, seed=42):
    """Percentile bootstrap CI for ROC AUC.

    Args:
        labels: 1d array of ground-truth 0/1
        preds:  1d array of predicted probabilities
        n_boot: number of bootstrap resamples
        ci:     confidence level (default 0.95)
        seed:   RNG seed
    Returns: (auc_point, lo, hi)
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    n = len(labels)
    if n < 2 or len(np.unique(labels)) < 2:
        return float('nan'), float('nan'), float('nan')

    point = roc_auc_score(labels, preds)
    rng = np.random.default_rng(seed)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            aucs[b] = roc_auc_score(labels[idx], preds[idx])
        except ValueError:
            # Resample contained only one class
            aucs[b] = np.nan
    aucs = aucs[~np.isnan(aucs)]
    alpha = (1 - ci) / 2
    lo, hi = np.quantile(aucs, [alpha, 1 - alpha])
    return float(point), float(lo), float(hi)


def paired_bootstrap_p_value(labels, preds_a, preds_b, n_boot=1000, seed=42):
    """One-sided paired bootstrap test: H0 = AUC(B) >= AUC(A).

    Returns p-value for "model A is better than model B".
    Useful for the headline DKT-beats-LLM comparison.
    """
    labels = np.asarray(labels)
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    n = len(labels)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            d = (roc_auc_score(labels[idx], preds_a[idx])
                 - roc_auc_score(labels[idx], preds_b[idx]))
        except ValueError:
            d = np.nan
        diffs[b] = d
    diffs = diffs[~np.isnan(diffs)]
    return float((diffs <= 0).mean())


def summarize_metrics(labels, preds, auc=None, acc=None, n_boot=500):
    """Compute AUC, ACC, AUC bootstrap CI in one call.

    If labels/preds are None, accept precomputed (auc, acc) (used when an
    underlying model returns scalar metrics only).
    """
    if labels is None and auc is not None:
        return {'AUC': float(auc), 'ACC': float(acc),
                'AUC_CI_lo': None, 'AUC_CI_hi': None,
                'n_predictions': None}
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    auc_pt, lo, hi = bootstrap_auc_ci(labels, preds, n_boot=n_boot)
    return {
        'AUC': auc_pt,
        'AUC_CI_lo': lo,
        'AUC_CI_hi': hi,
        'ACC': float(accuracy_score(labels, (preds > 0.5).astype(int))),
        'n_predictions': int(len(labels)),
    }


def per_step_auc(labels, preds, steps, max_step=20):
    """AUC stratified by position within student sequence.

    Useful for the "early sequence error" analysis from the FoundationalASSIST
    paper — LLM fine-tuning improves average AUC but stays worse than DKT
    precisely on the early-sequence predictions that matter most for adaptive
    tutoring.
    """
    labels = np.asarray(labels)
    preds = np.asarray(preds)
    steps = np.asarray(steps)
    out = {}
    for s in range(1, max_step + 1):
        m = steps == s
        if m.sum() < 20 or len(np.unique(labels[m])) < 2:
            continue
        out[s] = float(roc_auc_score(labels[m], preds[m]))
    return out
