"""
Regenerate results/benchmark_results.png from results/kt_results.json.
Adds 95% bootstrap CI error bars when available.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results' / 'kt_results.json'
OUT = ROOT / 'results' / 'benchmark_results.png'

# Color scheme: KT models in one tone, LLMs in another, baselines grey
COLORS = {
    'Trivial':                  '#9ca3af',
    'BKT':                      '#3b82f6',
    'DKT':                      '#1e40af',
    'SAKT':                     '#1e40af',
    'AKT':                      '#1e3a8a',
    'LLM_zero_shot':            '#dc2626',  # Anthropic Haiku zero
    'LLM_few_shot':             '#b91c1c',  # Anthropic Haiku few
    'LLM_cot':                  '#7f1d1d',
    'LLM_zero_shot_gpt4omini':  '#f97316',  # OpenAI gpt-4o-mini zero
    'LLM_few_shot_gpt4omini':   '#ea580c',  # OpenAI gpt-4o-mini few
}

DISPLAY = {
    'Trivial': 'Trivial',
    'BKT': 'BKT',
    'DKT': 'DKT (LSTM)',
    'SAKT': 'SAKT',
    'AKT': 'AKT',
    'LLM_zero_shot':            'Haiku 4.5\n0-shot',
    'LLM_few_shot':             'Haiku 4.5\nfew-shot',
    'LLM_cot':                  'Haiku 4.5\nCoT',
    'LLM_zero_shot_gpt4omini':  'GPT-4o-mini\n0-shot',
    'LLM_few_shot_gpt4omini':   'GPT-4o-mini\nfew-shot',
}


def main():
    with open(RESULTS) as f:
        results = json.load(f)

    models = list(results.keys())
    aucs = [results[m]['AUC'] for m in models]
    accs = [results[m]['ACC'] for m in models]

    # Error bars from CI when available
    auc_lo = []
    auc_hi = []
    for m in models:
        r = results[m]
        if r.get('AUC_CI_lo') is not None:
            auc_lo.append(r['AUC'] - r['AUC_CI_lo'])
            auc_hi.append(r['AUC_CI_hi'] - r['AUC'])
        else:
            auc_lo.append(0)
            auc_hi.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(models))
    bar_colors = [COLORS.get(m, '#6b7280') for m in models]
    labels = [DISPLAY.get(m, m) for m in models]

    # AUC plot with error bars
    axes[0].bar(x, aucs, yerr=[auc_lo, auc_hi], capsize=4,
                color=bar_colors, edgecolor='black', linewidth=0.5)
    axes[0].axhline(0.5, color='black', ls=':', lw=1, label='Random')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_ylabel('Test AUC')
    axes[0].set_title('Test AUC (95% bootstrap CI where available)')
    axes[0].set_ylim(0.45, max(0.85, max(aucs) + 0.05))
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].legend(loc='upper left')

    # Annotate bars with values
    for xi, v in zip(x, aucs):
        axes[0].text(xi, v + 0.005, f'{v:.3f}', ha='center', fontsize=9)

    # ACC plot
    axes[1].bar(x, accs, color=bar_colors, edgecolor='black', linewidth=0.5)
    base_rate = max(accs[0], 0.5)
    axes[1].axhline(base_rate, color='black', ls=':', lw=1,
                    label=f'Majority class ({base_rate:.3f})')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_ylabel('Test Accuracy')
    axes[1].set_title('Test Accuracy')
    axes[1].set_ylim(0.65, max(0.78, max(accs) + 0.02))
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].legend(loc='upper left')

    for xi, v in zip(x, accs):
        axes[1].text(xi, v + 0.002, f'{v:.3f}', ha='center', fontsize=9)

    fig.suptitle('Knowledge Tracing Benchmark: Specialized KT vs. LLMs',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {OUT}")
    return OUT


if __name__ == '__main__':
    main()
