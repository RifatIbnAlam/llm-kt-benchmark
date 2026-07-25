"""
Main benchmark runner.
Trains and evaluates: Trivial, BKT, DKT, SAKT, AKT, LLM (zero-shot), LLM (few-shot)
Saves results to results/kt_results.json and results/results_table.csv

Usage:
    cd kt_benchmark
    python run_benchmark.py                  # full run
    python run_benchmark.py --skip-llm       # skip API-calling LLM step
    python run_benchmark.py --models bkt,dkt # subset of models
    python run_benchmark.py --quick          # tiny sample for smoke testing
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Anchor all paths to this script's location — no hardcoded /home/claude prefixes
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from utils.metrics import bootstrap_auc_ci, summarize_metrics  # noqa: E402


def load_data(quick=False):
    train_df = pd.read_csv(ROOT / 'data' / 'train.csv')
    val_df   = pd.read_csv(ROOT / 'data' / 'val.csv')
    test_df  = pd.read_csv(ROOT / 'data' / 'test.csv')
    if quick:
        # Take first 100 students from each split for smoke testing
        def _sub(df):
            uids = df['user_id'].unique()[:100]
            return df[df['user_id'].isin(uids)].copy()
        train_df, val_df, test_df = map(_sub, [train_df, val_df, test_df])
    return train_df, val_df, test_df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--models', default='trivial,bkt,dkt,sakt,akt,llm_zero,llm_few',
                   help='comma-separated list of models to run')
    p.add_argument('--skip-llm', action='store_true',
                   help='skip LLM models (no API calls)')
    p.add_argument('--quick', action='store_true',
                   help='use a tiny subset for smoke testing')
    p.add_argument('--llm-students', type=int, default=100,
                   help='number of students to sample for LLM evaluation')
    p.add_argument('--llm-model', default='claude-sonnet-4-5-20250929',
                   help='LLM model id (anthropic)')
    p.add_argument('--n-skills', type=int, default=110)
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print("="*60)
    print("Loading data...")
    train_df, val_df, test_df = load_data(quick=args.quick)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"N skills: {args.n_skills}")

    requested = set(args.models.split(','))
    if args.skip_llm:
        requested -= {'llm_zero', 'llm_few'}

    results = {}
    raw_predictions = {}  # for downstream bootstrap CI / per-step error analysis

    # Incremental save helper — call after each model so a late crash
    # doesn't lose earlier results
    def _save():
        os.makedirs(ROOT / 'results', exist_ok=True)
        results_path = ROOT / 'results' / 'kt_results.json'
        existing = {}
        if results_path.exists():
            try:
                with open(results_path) as f:
                    existing = json.load(f)
            except Exception:
                existing = {}
        existing.update(results)
        with open(results_path, 'w') as f:
            json.dump(existing, f, indent=2)

    # ── Trivial ──────────────────────────────────────────────────────────────
    if 'trivial' in requested:
        print("\n" + "="*60)
        print("[Trivial] always predict mean correct rate")
        from models.llm_kt import TrivialBaseline
        tb = TrivialBaseline()
        tb.fit(train_df)
        t0 = time.time()
        labels = test_df['correct'].values.astype(int)
        preds = np.full(len(labels), tb.p)
        m = summarize_metrics(labels, preds)
        m['time_s'] = round(time.time() - t0, 2)
        results['Trivial'] = m
        raw_predictions['Trivial'] = (preds, labels)
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f}")

    _save()

    # ── BKT ──────────────────────────────────────────────────────────────────
    if 'bkt' in requested:
        print("\n" + "="*60)
        print("[BKT] Bayesian Knowledge Tracing")
        from models.bkt import BKT
        bkt = BKT(n_skills=args.n_skills, max_iter=30)
        t0 = time.time()
        bkt.fit(train_df)
        preds = bkt.predict(test_df)
        labels = test_df.sort_values(['user_id', 'order'])['correct'].values
        m = summarize_metrics(labels, preds)
        m['time_s'] = round(time.time() - t0, 2)
        results['BKT'] = m
        raw_predictions['BKT'] = (preds, labels)
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f} | time={m['time_s']}s")

    _save()

    # ── DKT ──────────────────────────────────────────────────────────────────
    if 'dkt' in requested:
        print("\n" + "="*60)
        print("[DKT] LSTM-based Deep Knowledge Tracing")
        from models.dkt import DKT
        dkt = DKT(n_skills=args.n_skills, emb_dim=64, hidden_dim=128,
                  lr=1e-3, batch_size=args.batch_size, epochs=args.epochs)
        t0 = time.time()
        dkt.fit(train_df, val_df)
        preds, labels = dkt.predict(test_df)
        m = summarize_metrics(labels, preds)
        m['time_s'] = round(time.time() - t0, 2)
        results['DKT'] = m
        raw_predictions['DKT'] = (preds, labels)
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f} | time={m['time_s']}s")

    _save()

    # ── SAKT ─────────────────────────────────────────────────────────────────
    if 'sakt' in requested:
        print("\n" + "="*60)
        print("[SAKT] Self-Attentive Knowledge Tracing")
        from models.sakt import SAKT
        sakt = SAKT(n_skills=args.n_skills, d_model=128, n_heads=4, n_blocks=2,
                    lr=1e-3, batch_size=args.batch_size, epochs=args.epochs)
        t0 = time.time()
        sakt.fit(train_df, val_df)
        sak_results = sakt.evaluate(test_df)
        m = summarize_metrics(sak_results['labels'], sak_results['preds'])
        m['time_s'] = round(time.time() - t0, 2)
        results['SAKT'] = m
        raw_predictions['SAKT'] = (sak_results['preds'], sak_results['labels'])
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f} | time={m['time_s']}s")

    _save()

    # ── AKT ──────────────────────────────────────────────────────────────────
    if 'akt' in requested:
        print("\n" + "="*60)
        print("[AKT] Attentive KT (Ghosh+ 2020) — strongest specialized baseline")
        from models.akt import AKT
        akt = AKT(n_skills=args.n_skills, d_model=128, n_heads=4, n_blocks=2,
                  lr=1e-3, batch_size=args.batch_size, epochs=args.epochs)
        t0 = time.time()
        akt.fit(train_df, val_df)
        akt_results = akt.evaluate(test_df)
        m = summarize_metrics(akt_results['labels'], akt_results['preds'])
        m['time_s'] = round(time.time() - t0, 2)
        results['AKT'] = m
        raw_predictions['AKT'] = (akt_results['preds'], akt_results['labels'])
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f} | time={m['time_s']}s")

    _save()

    # ── LLM zero-shot ────────────────────────────────────────────────────────
    if 'llm_zero' in requested:
        print("\n" + "="*60)
        print(f"[LLM zero-shot] {args.llm_model}")
        from models.llm_kt import LLMKnowledgeTracer
        tracer = LLMKnowledgeTracer(mode='zero_shot',
                                    max_students=args.llm_students,
                                    model=args.llm_model)
        t0 = time.time()
        r = tracer.evaluate(test_df)
        m = summarize_metrics(r['labels'], r['preds'])
        m['time_s'] = round(time.time() - t0, 2)
        m['api_calls'] = r['api_calls']
        m['est_cost_usd'] = r['est_cost_usd']
        results['LLM_zero_shot'] = m
        raw_predictions['LLM_zero_shot'] = (r['preds'], r['labels'])
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f} | calls={m['api_calls']} | "
              f"cost=${m['est_cost_usd']}")

    _save()

    # ── LLM few-shot ─────────────────────────────────────────────────────────
    if 'llm_few' in requested:
        print("\n" + "="*60)
        print(f"[LLM few-shot] {args.llm_model}")
        from models.llm_kt import LLMKnowledgeTracer
        tracer = LLMKnowledgeTracer(mode='few_shot',
                                    max_students=args.llm_students,
                                    model=args.llm_model)
        t0 = time.time()
        r = tracer.evaluate(test_df)
        m = summarize_metrics(r['labels'], r['preds'])
        m['time_s'] = round(time.time() - t0, 2)
        m['api_calls'] = r['api_calls']
        m['est_cost_usd'] = r['est_cost_usd']
        results['LLM_few_shot'] = m
        raw_predictions['LLM_few_shot'] = (r['preds'], r['labels'])
        print(f"  AUC={m['AUC']:.4f} CI[{m['AUC_CI_lo']:.4f}, {m['AUC_CI_hi']:.4f}] "
              f"| ACC={m['ACC']:.4f} | calls={m['api_calls']} | "
              f"cost=${m['est_cost_usd']}")

    _save()

    # ── Save (merge with existing results, don't clobber unrun models) ──────
    os.makedirs(ROOT / 'results', exist_ok=True)
    results_path = ROOT / 'results' / 'kt_results.json'
    existing = {}
    if results_path.exists():
        try:
            with open(results_path) as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update(results)  # new results overwrite old for the same model
    with open(results_path, 'w') as f:
        json.dump(existing, f, indent=2)
    results = existing  # so the summary table prints all models

    # Raw predictions for downstream analysis (per-step error, calibration, etc.)
    # Merge with whatever was already on disk so a partial run (e.g. --models sakt)
    # doesn't wipe out raw predictions saved by earlier runs of other models.
    npz_path = ROOT / 'results' / 'raw_predictions.npz'
    existing_npz = {}
    if npz_path.exists():
        try:
            with np.load(npz_path) as npz_file:
                existing_npz = {k: npz_file[k] for k in npz_file.files}
        except Exception:
            existing_npz = {}
    existing_npz.update({k: np.stack([p, l]) for k, (p, l) in raw_predictions.items()
                          if p is not None and len(p) > 0})
    np.savez_compressed(npz_path, **existing_npz)

    # Summary table
    print("\n" + "="*60)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*60)
    print(f"{'Model':<18} {'AUC':>7} {'95% CI':>17} {'ACC':>7} {'Time':>8}")
    print("-"*64)
    for model, r in results.items():
        lo, hi = r.get('AUC_CI_lo'), r.get('AUC_CI_hi')
        ci = f"[{lo:.3f},{hi:.3f}]" if (lo is not None and hi is not None) else '       -       '
        t = f"{r.get('time_s', '-')}s" if r.get('time_s') else '-'
        print(f"{model:<18} {r['AUC']:>7.4f} {ci:>17} {r['ACC']:>7.4f} {t:>8}")

    rows = [{'Model': k, **{kk: vv for kk, vv in v.items()}} for k, v in results.items()]
    pd.DataFrame(rows).to_csv(ROOT / 'results' / 'results_table.csv', index=False)
    print(f"\nResults saved to {ROOT / 'results'}")


if __name__ == '__main__':
    main()
