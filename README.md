# KT Benchmark: LLMs vs. Specialized Knowledge Tracing Models

A reproducible benchmark comparing frontier LLMs (Anthropic Claude Haiku 4.5, OpenAI GPT-4o-mini)
against four specialized Knowledge Tracing models on student performance prediction, using the
real ASSISTments-2009 corrected dataset. 
## Project structure

```
kt_benchmark/
├── data/
│   ├── skill_builder_data_corrected.csv   # Real ASSISTments-2009 corrected raw data
│   ├── train.csv  val.csv  test.csv       # 70/10/20 student-level splits (real data)
│   └── assist09_synthetic.csv, assist09_real.csv  # earlier synthetic/staging data, superseded
├── scripts/
│   └── prepare_real_data.py    # Filtering/splitting pipeline (see paper_draft.md §3.1)
├── models/
│   ├── bkt.py           # Bayesian Knowledge Tracing (EM)
│   ├── dkt.py            # Deep Knowledge Tracing (LSTM)
│   ├── sakt.py           # Self-Attentive KT (Transformer)
│   ├── akt.py             # Attentive KT (Ghosh+ 2020) — strongest specialized baseline
│   └── llm_kt.py          # Multi-provider LLM KT (Anthropic / OpenAI / DeepSeek)
├── utils/
│   └── metrics.py        # Bootstrap CI + paired bootstrap p-value
├── results/
│   ├── kt_results.json
│   ├── results_table.csv
│   ├── benchmark_results.png
│   ├── figure_4_2_*.png              # calibration, cost frontier, distributions (paper §4.2)
│   ├── llm_predictions_*.jsonl       # resumable per-call checkpoints
│   └── raw_predictions.npz           # raw (preds, labels) for downstream analysis
├── make_figure.py                    # Regenerates results/benchmark_results.png from kt_results.json
├── make_cost_frontier.py             # Regenerates results/figure_4_2_cost_frontier.png from Table 2
├── make_calibration_steps_figure.py  # Regenerates results/figure_4_2_calibration_and_steps.png from llm_predictions_*.jsonl
├── make_distributions_figure.py      # Regenerates results/figure_4_2_distributions.png from llm_predictions_*.jsonl
├── run_benchmark.py          # Main orchestrator
├── paper_draft.md            # Full manuscript draft
└── submission/                # cover_letter.docx, manuscript.docx, title_page.docx
```

## Current results (real ASSISTments-2009 corrected)

208,644 interactions after preprocessing, 2,763 students, 101 skills, 70/10/20 student-level split.
See `paper_draft.md` §3.1 for the exact filtering pipeline.

| Model                  | AUC    | 95% CI            | ACC    | n_pred  | Source                    |
|-------------------------|--------|-------------------|--------|---------|---------------------------|
| Trivial                 | 0.500  | —                 | 0.653  | 43,763  | always-mean baseline      |
| BKT                     | 0.678  | [0.673, 0.684]    | 0.698  | 43,763  | EM trained                |
| DKT                     | 0.763  | [0.757, 0.768]    | 0.730  | 32,946  | 15 epochs, LSTM           |
| SAKT                    | 0.722  | [0.716, 0.728]    | 0.701  | 32,946  | 15 epochs, transformer    |
| AKT                     | 0.748  | [0.742, 0.754]    | 0.717  | 32,946  | 30 epochs, transformer    |
| Haiku 4.5 zero-shot     | 0.696  | [0.670, 0.723]    | 0.681  | 1,690   | real API calls            |
| Haiku 4.5 few-shot      | 0.703  | [0.678, 0.730]    | 0.667  | 1,650   | real API calls            |
| GPT-4o-mini zero-shot   | 0.709  | [0.684, 0.735]    | 0.654  | 1,690   | real API calls            |
| GPT-4o-mini few-shot    | 0.711  | [0.686, 0.738]    | 0.691  | 1,650   | real API calls            |

The headline gap: **DKT beats the best LLM condition (GPT-4o-mini few-shot) by 0.052 AUC**
(non-overlapping 95% CIs, p < 0.001 by paired bootstrap test). All four LLM conditions cluster
within 0.015 AUC of one another, invariant to provider and prompt strategy.

**Cost.** LLM API calls cost $0.03–$0.40 per 1,000 predictions ($126–$1,580/year at a
4M-prediction deployment); specialized models run locally at $0. Full benchmark cost $1.28 in
real API charges. (Note: an earlier version of `models/llm_kt.py` billed all Anthropic calls at
Claude Sonnet 4.5 rates regardless of the model actually used — since this benchmark used Claude
Haiku 4.5, that inflated the logged Anthropic costs 3x. The bug is fixed; figures above and in
`paper_draft.md` reflect the corrected, model-aware pricing.)

## Installation

```bash
pip install -r requirements.txt
```

## Data setup

The raw ASSISTments-2009 corrected CSV is not committed to this repo (large file, and
ASSISTments requires registration to redistribute). To reproduce from scratch:

1. Register and download `skill_builder_data_corrected.csv` from
   <https://sites.google.com/site/assistmentsdata/> into `data/`.
2. Run the preprocessing pipeline (filtering, splitting — see `paper_draft.md` §3.1):
   ```bash
   python scripts/prepare_real_data.py
   ```
   This produces `data/assist09_real.csv` and the `data/{train,val,test}.csv` splits used
   by every other command below.

## Running

### Full benchmark
```bash
cd kt_benchmark
python run_benchmark.py
```

### Subset
```bash
python run_benchmark.py --models bkt,dkt,akt    # specialized only
python run_benchmark.py --skip-llm              # everything except LLMs
python run_benchmark.py --quick                 # tiny smoke test
```

### LLM experiments (require API keys)

Set whichever provider keys you have:

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export DEEPSEEK_API_KEY=...

# Anthropic Claude Haiku 4.5 (the model used to produce the results above)
python run_benchmark.py --models llm_zero,llm_few \
    --llm-students 100 --llm-model claude-haiku-4-5-20251001

# OpenAI GPT-4o-mini
python -c "
from models.llm_kt import LLMKnowledgeTracer
import pandas as pd
df = pd.read_csv('data/test.csv')
t = LLMKnowledgeTracer(mode='zero_shot', provider='openai',
                      model='gpt-4o-mini', max_students=100)
print(t.evaluate(df))
"
```

Predictions are checkpointed to `results/llm_predictions_{provider}_{mode}.jsonl`
after every API call, so interrupted runs resume automatically.

## Known limitations / open items before submission

**Reproducibility:** `run_benchmark.py` seeds NumPy (`np.random.seed`) but not PyTorch, so
   DKT/SAKT/AKT results vary slightly (~0.001–0.002 AUC) between runs of the same seed. Add
   `torch.manual_seed(args.seed)` in `main()` if bit-exact reproduction is required.

