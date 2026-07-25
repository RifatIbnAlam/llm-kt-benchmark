"""
Regenerate results/figure_4_2_calibration_and_steps.png from the raw
results/llm_predictions_*.jsonl checkpoints (paper_draft.md 4.2.1-4.2.2).

Left panel: per-position AUC pooled across all four LLM conditions,
+/-1-step smoothing window (Fig. 2 left, section 4.2.1).
Right panel: reliability diagram, 10 equal-width bins, all four
LLM conditions (Fig. 2 right, section 4.2.2).

Reconstructed because the original figure predates this repo's git
history and no generation script was available; the position-AUC
values (0.712 / 0.722 / 0.689 / 0.697 at positions 1/5/10/20) and ECE
values were verified against paper_draft.md before trusting this script.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'figure_4_2_calibration_and_steps.png'
WARMUP = 5
DKT_AUC = 0.763

FILES = {
    'Haiku 4.5 zero-shot':   'llm_predictions_anthropic_zero_shot.jsonl',
    'Haiku 4.5 few-shot':    'llm_predictions_anthropic_few_shot.jsonl',
    'GPT-4o-mini zero-shot': 'llm_predictions_openai_zero_shot.jsonl',
    'GPT-4o-mini few-shot':  'llm_predictions_openai_few_shot.jsonl',
}
COLORS = {
    'Haiku 4.5 zero-shot':   '#dc2626',
    'Haiku 4.5 few-shot':    '#b91c1c',
    'GPT-4o-mini zero-shot': '#f97316',
    'GPT-4o-mini few-shot':  '#ea580c',
}


def load_all():
    per_condition = {}
    pooled = []
    for name, fn in FILES.items():
        rows = [json.loads(l) for l in open(RESULTS / fn)]
        for r in rows:
            r['position'] = r['t'] - WARMUP + 1
        per_condition[name] = rows
        pooled.extend(rows)
    return per_condition, pooled


def pos_auc(pooled, p, window=1):
    positions = np.array([r['position'] for r in pooled])
    labels = np.array([r['label'] for r in pooled])
    preds = np.array([r['prob'] for r in pooled])
    m = (positions >= p - window) & (positions <= p + window)
    if len(np.unique(labels[m])) < 2:
        return np.nan
    return roc_auc_score(labels[m], preds[m])


def reliability_bins(preds, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (preds >= lo) & (preds < hi if i < n_bins - 1 else preds <= hi)
        if m.sum() == 0:
            continue
        xs.append(preds[m].mean())
        ys.append(labels[m].mean())
    return xs, ys


def main():
    per_condition, pooled = load_all()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Left: per-position AUC, +/-1-step smoothing
    positions = list(range(1, 21))
    aucs = [pos_auc(pooled, p) for p in positions]
    axes[0].plot(positions, aucs, marker='o', color='#1e40af', lw=1.5,
                 label='Pooled LLM AUC (±1-step smoothing)')
    axes[0].axhline(DKT_AUC, color='#1e40af', ls='--', lw=1,
                     label=f"DKT test AUC ({DKT_AUC})")
    axes[0].set_xlabel('Position in evaluation sequence')
    axes[0].set_ylabel('AUC')
    axes[0].set_title('Per-position AUC (pooled across 4 LLM conditions)')
    axes[0].set_xticks(range(1, 21, 2))
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc='lower right', fontsize=8)

    # Right: reliability diagram, all 4 conditions
    axes[1].plot([0, 1], [0, 1], ls=':', color='black', lw=1, label='Perfect calibration')
    for name, rows in per_condition.items():
        preds = np.array([r['prob'] for r in rows])
        labels = np.array([r['label'] for r in rows])
        xs, ys = reliability_bins(preds, labels)
        axes[1].plot(xs, ys, marker='o', ms=4, lw=1.2, color=COLORS[name], label=name)
    axes[1].set_xlabel('Mean predicted probability (bin)')
    axes[1].set_ylabel('Empirical correct rate (bin)')
    axes[1].set_title('Reliability diagram (10 equal-width bins)')
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc='upper left', fontsize=7)

    plt.tight_layout()
    plt.savefig(OUT, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {OUT}")


if __name__ == '__main__':
    main()
