"""
Convert raw ASSISTments-2009 skill_builder_data_corrected.csv to our schema and
regenerate train/val/test student-level splits.

Standard preprocessing (matches pyKT, DKT-original, AKT, and most recent papers):
  1. Drop rows with missing skill_id
  2. Keep only original problems (`original == 1`) — drop scaffolding interactions
  3. For multi-skill rows ("skill_id = 27,28,29"), take the first listed skill
  4. Drop students with fewer than 10 interactions
  5. Re-encode skill_id to a dense [0, n_skills) range
  6. Add a per-student monotonic `order` column (0-indexed)
  7. Save to data/assist09_real.csv
  8. Student-level 70/10/20 split into data/{train,val,test}.csv

Usage:
    python scripts/prepare_real_data.py
    # then update n_skills if the printed count differs from 110:
    python run_benchmark.py --skip-llm --n-skills <printed_value>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
RAW_PATH = DATA / 'skill_builder_data_corrected.csv'
OUT_PATH = DATA / 'assist09_real.csv'
SEED = 42
MIN_INTERACTIONS = 10


def load_raw():
    print(f"Loading {RAW_PATH} ...")
    # ASSIST-2009 has odd encoding sometimes; use latin-1 as a fallback
    try:
        df = pd.read_csv(RAW_PATH, low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(RAW_PATH, low_memory=False, encoding='latin-1')
    print(f"  raw rows: {len(df):,}  cols: {len(df.columns)}")
    return df


def preprocess(df):
    # Required columns
    needed = ['order_id', 'user_id', 'skill_id', 'correct', 'original']
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Got: {list(df.columns)[:30]}")

    # Drop rows with missing skill_id
    before = len(df)
    df = df.dropna(subset=['skill_id'])
    print(f"  drop NaN skill_id: {before:,} -> {len(df):,}")

    # Keep only original problems (drop scaffolding/hint interactions)
    before = len(df)
    df = df[df['original'] == 1]
    print(f"  keep original==1:  {before:,} -> {len(df):,}")

    # Multi-skill rows like "27,28,29" -> take first skill
    df['skill_id'] = df['skill_id'].astype(str).str.split(',').str[0]
    df['skill_id'] = pd.to_numeric(df['skill_id'], errors='coerce')
    before = len(df)
    df = df.dropna(subset=['skill_id'])
    df['skill_id'] = df['skill_id'].astype(int)
    print(f"  resolve multi-skill: {before:,} -> {len(df):,}")

    # Coerce correct to {0,1}
    df['correct'] = pd.to_numeric(df['correct'], errors='coerce')
    df = df.dropna(subset=['correct'])
    df['correct'] = df['correct'].astype(int).clip(0, 1)

    # Drop students with < MIN_INTERACTIONS
    counts = df.groupby('user_id').size()
    keep_users = counts[counts >= MIN_INTERACTIONS].index
    before = len(df)
    df = df[df['user_id'].isin(keep_users)]
    print(f"  drop short users (<{MIN_INTERACTIONS}): {before:,} -> {len(df):,}")

    # Re-encode skill_id to dense [0, n_skills)
    skill_codes = {s: i for i, s in enumerate(sorted(df['skill_id'].unique()))}
    df['skill_id'] = df['skill_id'].map(skill_codes)

    # Sort by user, then order_id; add per-user monotonic 'order'
    df = df.sort_values(['user_id', 'order_id']).reset_index(drop=True)
    df['order'] = df.groupby('user_id').cumcount()

    keep_cols = ['user_id', 'skill_id', 'correct', 'order', 'order_id', 'problem_id']
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    n_skills = df['skill_id'].nunique()
    n_students = df['user_id'].nunique()
    p_correct = df['correct'].mean()
    print(f"\n  Final: {len(df):,} rows | {n_students:,} students | "
          f"{n_skills} skills | mean(correct)={p_correct:.4f}")
    return df, n_skills


def split_students(df, train_frac=0.7, val_frac=0.1, seed=SEED):
    """Student-level split: a student appears in exactly one of train/val/test."""
    users = np.array(sorted(df['user_id'].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(users)
    n = len(users)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_u = set(users[:n_train])
    val_u = set(users[n_train:n_train + n_val])
    test_u = set(users[n_train + n_val:])

    train = df[df['user_id'].isin(train_u)].copy()
    val   = df[df['user_id'].isin(val_u)].copy()
    test  = df[df['user_id'].isin(test_u)].copy()
    print(f"\n  Split: train={len(train):,} ({len(train_u)} users) | "
          f"val={len(val):,} ({len(val_u)}) | test={len(test):,} ({len(test_u)})")
    return train, val, test


def main():
    df = load_raw()
    df, n_skills = preprocess(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved cleaned data: {OUT_PATH}")

    train, val, test = split_students(df)
    train.to_csv(DATA / 'train.csv', index=False)
    val.to_csv(DATA / 'val.csv', index=False)
    test.to_csv(DATA / 'test.csv', index=False)
    print(f"Saved splits to {DATA}/")

    print("\n" + "="*60)
    print(f"NEXT STEPS")
    print("="*60)
    print(f"Number of skills: {n_skills}")
    if n_skills != 110:
        print(f"\nNote: real data has {n_skills} skills, not 110.")
        print(f"Run benchmark with: python run_benchmark.py --skip-llm "
              f"--n-skills {n_skills}")
    else:
        print(f"\nRun: python run_benchmark.py --skip-llm")


if __name__ == '__main__':
    main()
