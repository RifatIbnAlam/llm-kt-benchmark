"""
Regenerate results/figure_4_2_cost_frontier.png from Table 2 (paper_draft.md §4.2.5).
Cost-accuracy frontier: Test AUC vs. cost per 1,000 predictions.
Specialized KT models run locally at zero marginal cost; LLM costs are
computed from logged est_cost_usd / api_calls (see results/kt_results.json).
"""
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'figure_4_2_cost_frontier.png'

# (label, AUC, cost per 1,000 predictions in USD, is_specialized)
POINTS = [
    ('BKT',                  0.678, 0.0,   True),
    ('DKT',                  0.763, 0.0,   True),
    ('SAKT',                 0.722, 0.0,   True),
    ('AKT',                  0.748, 0.0,   True),
    ('GPT-4o-mini\n0-shot',  0.709, 0.031, False),
    ('GPT-4o-mini\nfew-shot',0.711, 0.046, False),
    ('Haiku 4.5\n0-shot',    0.696, 0.312, False),
    ('Haiku 4.5\nfew-shot',  0.703, 0.395, False),
]

# Zero-cost points can't be shown on a log axis; place them at a small
# nominal cost floor purely for visualization and annotate as "free (local)".
FLOOR = 0.01

fig, ax = plt.subplots(figsize=(7, 5.5))
for label, auc, cost, is_kt in POINTS:
    x = max(cost, FLOOR)
    color = '#1e40af' if is_kt else '#dc2626'
    marker = 'o' if is_kt else '^'
    ax.scatter(x, auc, s=90, color=color, marker=marker,
               edgecolor='black', linewidth=0.5, zorder=3)
    ax.annotate(label, (x, auc), textcoords='offset points',
                xytext=(6, -3), fontsize=8)

ax.set_xscale('log')
ax.set_xlabel('Cost per 1,000 predictions (USD, log scale)\nSpecialized KT models shown at cost floor — true cost is $0 (local inference)')
ax.set_ylabel('Test AUC')
ax.set_title('Cost-Accuracy Frontier: Specialized KT vs. LLMs')
ax.grid(alpha=0.3, which='both')

kt_handle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1e40af',
                        markeredgecolor='black', markersize=9, label='Specialized KT (free, local)')
llm_handle = plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='#dc2626',
                         markeredgecolor='black', markersize=9, label='LLM (API cost)')
ax.legend(handles=[kt_handle, llm_handle], loc='lower right')

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f"Figure saved to {OUT}")
