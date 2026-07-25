"""
Bayesian Knowledge Tracing (BKT) — Corbett & Anderson 1995
Parameters per skill: p_init, p_learn, p_slip, p_guess
Fitted via EM algorithm.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score


class BKT:
    def __init__(self, n_skills, max_iter=50, tol=1e-4):
        self.n_skills = n_skills
        self.max_iter = max_iter
        self.tol = tol
        # Per-skill parameters [n_skills]
        self.p_init  = np.full(n_skills, 0.3)
        self.p_learn = np.full(n_skills, 0.2)
        self.p_slip  = np.full(n_skills, 0.1)
        self.p_guess = np.full(n_skills, 0.2)

    def _clip(self, x):
        return np.clip(x, 1e-6, 1 - 1e-6)

    def _get_sequences(self, df):
        """Group df into per-(student, skill) sequences."""
        return df.groupby(['user_id', 'skill_id'])['correct'].apply(list)

    def fit(self, train_df):
        seqs = self._get_sequences(train_df)
        for iteration in range(self.max_iter):
            old_params = np.stack([self.p_init, self.p_learn, self.p_slip, self.p_guess])
            # Accumulate sufficient statistics per skill
            sum_pL0 = np.zeros(self.n_skills)
            sum_pT  = np.zeros(self.n_skills)
            sum_pS  = np.zeros(self.n_skills)
            sum_pG  = np.zeros(self.n_skills)
            cnt     = np.zeros(self.n_skills)

            for (uid, sk), obs in seqs.items():
                obs = np.array(obs)
                T = len(obs)
                if T < 2:
                    continue
                sk = int(sk)
                pi = self._clip(self.p_init[sk])
                pL = self._clip(self.p_learn[sk])
                pS = self._clip(self.p_slip[sk])
                pG = self._clip(self.p_guess[sk])

                # Forward pass
                alpha = np.zeros((T, 2))  # alpha[t, k]: P(obs_0..t, K_t=k)
                for t, y in enumerate(obs):
                    p_obs_given_k = np.array([
                        pG if y == 1 else (1 - pG),     # K=0: guess
                        (1 - pS) if y == 1 else pS       # K=1: 1-slip
                    ])
                    if t == 0:
                        alpha[t] = np.array([1 - pi, pi]) * p_obs_given_k
                    else:
                        # Transition: K_{t-1}=0 -> K_t=0: 1-pL; K_{t-1}=0 -> K_t=1: pL
                        #             K_{t-1}=1 -> K_t=0: 0;   K_{t-1}=1 -> K_t=1: 1
                        trans = np.array([
                            alpha[t-1, 0] * (1 - pL) + alpha[t-1, 1] * 0,
                            alpha[t-1, 0] * pL       + alpha[t-1, 1] * 1
                        ])
                        alpha[t] = trans * p_obs_given_k
                    alpha[t] /= (alpha[t].sum() + 1e-300)

                # Backward pass
                beta = np.zeros((T, 2))
                beta[-1] = 1.0
                for t in range(T - 2, -1, -1):
                    y_next = obs[t + 1]
                    p_obs_given_k = np.array([
                        pG if y_next == 1 else (1 - pG),
                        (1 - pS) if y_next == 1 else pS
                    ])
                    beta[t, 0] = ((1 - pL) * p_obs_given_k[0] * beta[t+1, 0] +
                                   pL      * p_obs_given_k[1] * beta[t+1, 1])
                    beta[t, 1] = (0       * p_obs_given_k[0] * beta[t+1, 0] +
                                   1      * p_obs_given_k[1] * beta[t+1, 1])
                    s = beta[t].sum() + 1e-300
                    beta[t] /= s

                # Posterior P(K_t=1 | obs)
                gamma = alpha * beta
                gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-300)

                # Accumulate stats
                sum_pL0[sk] += gamma[0, 1]
                cnt[sk]     += 1
                for t in range(T - 1):
                    sum_pT[sk] += gamma[t, 0]   # expected transitions from K=0
                for t, y in enumerate(obs):
                    if y == 0:
                        sum_pS[sk] += gamma[t, 1]
                    else:
                        sum_pG[sk] += gamma[t, 0]

            # M-step
            mask = cnt > 0
            self.p_init[mask]  = self._clip(sum_pL0[mask] / cnt[mask])
            denom_T = np.array([seqs.apply(len).groupby(level=1).sum().reindex(range(self.n_skills), fill_value=0).values], dtype=float).flatten()
            self.p_learn = self._clip(sum_pT / (denom_T + 1e-6))
            self.p_slip  = self._clip(sum_pS / (denom_T + 1e-6))
            self.p_guess = self._clip(sum_pG / (denom_T + 1e-6))

            # Check convergence
            new_params = np.stack([self.p_init, self.p_learn, self.p_slip, self.p_guess])
            delta = np.abs(new_params - old_params).max()
            if delta < self.tol:
                print(f"  BKT converged at iteration {iteration+1}")
                break

    def predict(self, test_df):
        """Return predicted P(correct) for each row."""
        preds = []
        for uid, group in test_df.groupby('user_id'):
            group = group.sort_values('order')
            # Track knowledge state per skill
            p_k = dict()
            for _, row in group.iterrows():
                sk = int(row['skill_id'])
                pi  = self.p_init[sk]
                pL  = self.p_learn[sk]
                pS  = self.p_slip[sk]
                pG  = self.p_guess[sk]

                p_know = p_k.get(sk, pi)
                p_correct = p_know * (1 - pS) + (1 - p_know) * pG
                preds.append(p_correct)

                # Update belief
                y = int(row['correct'])
                if y == 1:
                    num = p_know * (1 - pS)
                    den = p_correct
                else:
                    num = p_know * pS
                    den = 1 - p_correct
                p_know_given_obs = num / (den + 1e-9)
                p_k[sk] = p_know_given_obs * 1 + (1 - p_know_given_obs) * pL

        return np.array(preds)

    def evaluate(self, test_df):
        preds = self.predict(test_df)
        labels = test_df.sort_values(['user_id', 'order'])['correct'].values
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, (preds > 0.5).astype(int))
        return {'AUC': auc, 'ACC': acc}
