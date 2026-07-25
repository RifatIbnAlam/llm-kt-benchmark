"""
Regenerate results/figure_4_2_distributions.png from the raw
results/llm_predictions_*.jsonl checkpoints (paper_draft.md 4.2.3).

One probability histogram per LLM condition, showing anchoring on
round values (multiples of 0.05 / 0.10).

Reconstructed because the original figure predates this repo's git
history and no generation script was available; the unique-value
counts and top-5 shares (e.g. GPT-4o-mini few-shot: 29 unique values,
top five = 60% of predictions) were verified against paper_draft.md
before trusting this script.
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
OUT = RESULTS / 'figure_4_2_distributions.png'

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


def main():
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=False)

    for ax, (name, fn) in zip(axes, FILES.items()):
        rows = [json.loads(l) for l in open(RESULTS / fn)]
        probs = [r['prob'] for r in rows]
        n_unique = len(Counter(probs))

        ax.hist(probs, bins=np.arange(0, 1.02, 0.02), color=COLORS[name], edgecolor='none')
        ax.set_title(f"{name}\n({n_unique} unique values, n={len(probs)})", fontsize=9)
        ax.set_xlabel('Predicted probability')
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3, axis='y')

    axes[0].set_ylabel('Count')
    fig.suptitle('Predicted-probability distributions by LLM condition', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT, dpi=600, bbox_inches='tight')
    print(f"Figure saved to {OUT}")


if __name__ == '__main__':
    main()
