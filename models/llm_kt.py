"""
LLM-based Knowledge Tracing — multi-provider, resumable.

Providers
---------
  - anthropic: claude-sonnet/opus/haiku  (env: ANTHROPIC_API_KEY)
  - openai:    gpt-4o, gpt-4o-mini       (env: OPENAI_API_KEY)
  - deepseek:  deepseek-chat             (env: DEEPSEEK_API_KEY)

Modes
-----
  - zero_shot: history -> direct probability
  - few_shot:  history + 3 in-context examples -> probability
  - cot:       history + chain-of-thought -> probability

Resumability
------------
Predictions are checkpointed to results/llm_predictions_{provider}_{mode}.jsonl
after each call, so runs can be interrupted and continued. Re-running with the
same parameters skips students already covered.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


# ── Provider adapters ────────────────────────────────────────────────────────

class AnthropicProvider:
    name = 'anthropic'
    base_url = 'https://api.anthropic.com/v1/messages'

    # Per-MTok (input, output) pricing in USD. Keyed by substring match against
    # the model id, since Anthropic model strings embed a date suffix.
    PRICING = {
        'haiku':  (1, 5),    # Claude Haiku 4.5
        'sonnet': (3, 15),   # Claude Sonnet 4.5
        'opus':   (15, 75),  # Claude Opus 4.x (approx)
    }

    def __init__(self, model='claude-sonnet-4-5-20250929'):
        self.model = model
        self.api_key = os.environ.get('ANTHROPIC_API_KEY')

    def _rates(self):
        for key, rates in self.PRICING.items():
            if key in self.model.lower():
                return rates
        raise ValueError(f"No known pricing for Anthropic model '{self.model}'; "
                          f"add it to AnthropicProvider.PRICING")

    def call(self, prompt, max_tokens=128):
        import urllib.request
        import urllib.error
        import ssl
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
                "x-api-key": self.api_key,
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            data = json.loads(r.read())
        in_tok = data.get('usage', {}).get('input_tokens', 0)
        out_tok = data.get('usage', {}).get('output_tokens', 0)
        in_rate, out_rate = self._rates()
        cost = (in_tok * in_rate + out_tok * out_rate) / 1e6
        return data['content'][0]['text'], cost


class OpenAIProvider:
    name = 'openai'
    base_url = 'https://api.openai.com/v1/chat/completions'

    def __init__(self, model='gpt-4o-mini'):
        self.model = model
        self.api_key = os.environ.get('OPENAI_API_KEY')

    def call(self, prompt, max_tokens=128):
        import urllib.request
        import ssl
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        req = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
            data = json.loads(r.read())
        usage = data.get('usage', {})
        in_tok = usage.get('prompt_tokens', 0)
        out_tok = usage.get('completion_tokens', 0)
        # gpt-4o-mini approx pricing (Nov 2025): $0.15/$0.60 per MTok
        cost = (in_tok * 0.15 + out_tok * 0.60) / 1e6
        return data['choices'][0]['message']['content'], cost


class DeepSeekProvider(OpenAIProvider):
    """DeepSeek uses an OpenAI-compatible endpoint."""
    name = 'deepseek'
    base_url = 'https://api.deepseek.com/chat/completions'

    def __init__(self, model='deepseek-chat'):
        self.model = model
        self.api_key = os.environ.get('DEEPSEEK_API_KEY')

    def call(self, prompt, max_tokens=128):
        # Inherit OpenAIProvider.call but override pricing
        text, _ = super().call(prompt, max_tokens)
        # DeepSeek-chat: $0.07/$1.10 per MTok cached/output (approx Nov 2025)
        # Cost computed only for budget tracking; not exact
        return text, 0.0


def make_provider(provider, model=None):
    p = provider.lower()
    if p == 'anthropic':
        return AnthropicProvider(model or 'claude-sonnet-4-5-20250929')
    if p == 'openai':
        return OpenAIProvider(model or 'gpt-4o-mini')
    if p == 'deepseek':
        return DeepSeekProvider(model or 'deepseek-chat')
    raise ValueError(f"Unknown provider: {provider}")


# ── Prompts ──────────────────────────────────────────────────────────────────

def fmt_history(history_rows, max_n=None):
    rows = history_rows if max_n is None else history_rows.tail(max_n)
    return "\n".join(
        f"  - Skill {int(r.skill_id)}: {'correct' if r.correct == 1 else 'incorrect'}"
        for r in rows.itertuples()
    )


def zero_shot_prompt(history, target_skill):
    return (
        "You are an intelligent tutoring system analyzing a student's knowledge.\n\n"
        "Student's interaction history:\n"
        f"<history>\n{fmt_history(history)}\n</history>\n\n"
        f"Predict whether the student will answer a question on Skill {target_skill} correctly.\n"
        'Respond ONLY with JSON: {"probability": <float between 0 and 1>}'
    )


def few_shot_prompt(history, target_skill, n_history=15):
    return (
        "You are an intelligent tutoring system. Predict student performance.\n\n"
        "Examples:\n"
        '<example>\n  - Skill 5: correct\n  - Skill 5: correct\n'
        '  - Skill 12: incorrect\nPredict Skill 5: {"probability": 0.85}\n</example>\n'
        '<example>\n  - Skill 3: incorrect\n  - Skill 3: incorrect\n'
        '  - Skill 3: incorrect\nPredict Skill 3: {"probability": 0.15}\n</example>\n'
        '<example>\n  - Skill 7: correct\n  - Skill 2: incorrect\n'
        '  - Skill 7: correct\nPredict Skill 2: {"probability": 0.35}\n</example>\n\n'
        "Now predict for this student:\n"
        f"<history>\n{fmt_history(history, max_n=n_history)}\n</history>\n\n"
        f'Predict Skill {target_skill}. JSON only: {{"probability": <float>}}'
    )


def cot_prompt(history, target_skill, n_history=15):
    return (
        "You are analyzing a student's knowledge state.\n\n"
        f"<history>\n{fmt_history(history, max_n=n_history)}\n</history>\n\n"
        f"Step 1: count attempts on Skill {target_skill} in the history.\n"
        f"Step 2: assess overall student ability from other skills.\n"
        f"Step 3: predict P(correct) on Skill {target_skill}.\n\n"
        'After your reasoning, output a final line with JSON: {"probability": <float>}'
    )


PROMPT_BUILDERS = {
    'zero_shot': zero_shot_prompt,
    'few_shot':  few_shot_prompt,
    'cot':       cot_prompt,
}


# ── Parsing ──────────────────────────────────────────────────────────────────

PROB_REGEX = re.compile(r'"probability"\s*:\s*([0-9]*\.?[0-9]+)')


def parse_prob(text):
    """Extract probability in [0, 1]. Robust to truncation and extra text."""
    if not text:
        return 0.5
    # Try the last JSON object on the last line
    m = PROB_REGEX.findall(text)
    if m:
        try:
            return float(m[-1])
        except ValueError:
            pass
    # Last resort: any number between 0 and 1 in the text
    nums = re.findall(r'(?:^|[^0-9])(0?\.[0-9]+|0|1)(?:[^0-9]|$)', text)
    for n in nums[::-1]:
        try:
            v = float(n)
            if 0 <= v <= 1:
                return v
        except ValueError:
            continue
    return 0.5


# ── Main tracer ──────────────────────────────────────────────────────────────

class LLMKnowledgeTracer:
    def __init__(self, mode='zero_shot', n_history=15, max_students=200,
                 max_preds_per_student=20, warmup=5,
                 provider='anthropic', model=None,
                 sleep_s=0.3, checkpoint_dir=None):
        assert mode in PROMPT_BUILDERS
        self.mode = mode
        self.n_history = n_history
        self.max_students = max_students
        self.max_preds_per_student = max_preds_per_student
        self.warmup = warmup
        self.provider = make_provider(provider, model=model)
        self.sleep_s = sleep_s
        self.call_count = 0
        self.total_cost_estimate = 0.0
        self.failed_calls = 0

        self.checkpoint_dir = (Path(checkpoint_dir) if checkpoint_dir
                               else Path(__file__).resolve().parent.parent
                                    / 'results')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = (self.checkpoint_dir
                          / f"llm_predictions_{self.provider.name}_{mode}.jsonl")

    def _build_prompt(self, history, target_skill):
        builder = PROMPT_BUILDERS[self.mode]
        if self.mode == 'zero_shot':
            return builder(history, target_skill)
        return builder(history, target_skill, self.n_history)

    def _load_checkpoint(self):
        """Return set of (user_id, t) pairs already predicted."""
        seen = {}
        if not self.ckpt_path.exists():
            return seen
        with open(self.ckpt_path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    seen[(d['user_id'], d['t'])] = (d['prob'], d['label'])
                except Exception:
                    continue
        return seen

    def _append_checkpoint(self, user_id, t, target_skill, prob, label):
        with open(self.ckpt_path, 'a') as f:
            f.write(json.dumps({
                'user_id': int(user_id), 't': int(t),
                'skill': int(target_skill),
                'prob': float(prob), 'label': int(label),
            }) + '\n')

    def evaluate(self, test_df):
        students = test_df['user_id'].unique()
        if len(students) > self.max_students:
            np.random.seed(42)
            students = np.random.choice(students, self.max_students, replace=False)

        seen = self._load_checkpoint()
        if seen:
            print(f"  Resuming: {len(seen)} predictions already cached "
                  f"in {self.ckpt_path.name}")

        all_preds, all_labels = [], []
        # Add already-cached predictions for students in our sample
        student_set = set(students.tolist())
        for (uid, t), (prob, label) in seen.items():
            if uid in student_set:
                all_preds.append(prob)
                all_labels.append(label)

        for i, uid in enumerate(students):
            student_df = test_df[test_df.user_id == uid].sort_values('order')
            rows = list(student_df.itertuples())
            n_made = 0
            for t in range(self.warmup, len(rows)):
                if n_made >= self.max_preds_per_student:
                    break
                if (uid, t) in seen:
                    continue  # already in checkpoint
                history = student_df.iloc[:t]
                target = rows[t]
                target_skill = int(target.skill_id)
                label = int(target.correct)

                prompt = self._build_prompt(history, target_skill)
                try:
                    text, cost = self.provider.call(prompt)
                    self.total_cost_estimate += cost
                    self.call_count += 1
                    prob = parse_prob(text)
                except Exception as e:
                    self.failed_calls += 1
                    prob = 0.5
                prob = float(np.clip(prob, 0.01, 0.99))

                all_preds.append(prob)
                all_labels.append(label)
                self._append_checkpoint(uid, t, target_skill, prob, label)
                n_made += 1
                if self.sleep_s:
                    time.sleep(self.sleep_s)

            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(students)} students | "
                      f"calls={self.call_count} fail={self.failed_calls} | "
                      f"cost=${self.total_cost_estimate:.4f}")

        preds = np.array(all_preds)
        labels = np.array(all_labels)
        if len(preds) == 0 or len(np.unique(labels)) < 2:
            return {'AUC': float('nan'), 'ACC': float('nan'),
                    'preds': preds, 'labels': labels,
                    'api_calls': self.call_count, 'est_cost_usd': 0.0}
        return {
            'AUC': float(roc_auc_score(labels, preds)),
            'ACC': float(accuracy_score(labels, (preds > 0.5).astype(int))),
            'n_predictions': int(len(preds)),
            'preds': preds,
            'labels': labels,
            'api_calls': self.call_count,
            'failed_calls': self.failed_calls,
            'est_cost_usd': round(self.total_cost_estimate, 4),
        }


# ── Trivial baseline (kept here for backward compat with run_benchmark.py) ────

class TrivialBaseline:
    """Always predict the training-set correct rate."""

    def __init__(self, p=0.7):
        self.p = p

    def fit(self, train_df):
        self.p = float(train_df['correct'].mean())
        print(f"  Trivial baseline: always predict {self.p:.4f}")

    def evaluate(self, test_df):
        labels = test_df['correct'].values
        preds = np.full(len(labels), self.p)
        return {
            'AUC': float(roc_auc_score(labels, preds)),
            'ACC': float(accuracy_score(labels, (preds > 0.5).astype(int))),
        }
