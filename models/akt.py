"""
Attentive Knowledge Tracing (AKT) — Ghosh, Heffernan & Lan, KDD 2020
Context-aware self-attention with monotonic distance decay (Rasch + d_t).
Strongest specialized KT baseline; the right comparison point against LLMs.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from models.dkt import KTDataset, collate_fn


class MonotonicAttention(nn.Module):
    """Distance-aware self-attention from AKT.

    Score(q, k_i) = softmax(qK^T / sqrt(d_k) - gamma * d(t, i))
    where d(t, i) is exponentially-decayed temporal distance.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        # Per-head learnable decay parameter (clamped >= 0)
        self.gamma = nn.Parameter(torch.zeros(n_heads))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in, k_in, v_in, causal_mask):
        B, T, _ = q_in.shape

        def split(x):
            return x.view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        Q = split(self.q(q_in))
        K = split(self.k(k_in))
        V = split(self.v(v_in))

        # Standard scaled dot-product
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # [B, H, T, T]

        # Monotonic distance term: positions i,j with j>i are masked anyway;
        # for j<=i, distance |i-j| weighted by gamma_h
        idx = torch.arange(T, device=q_in.device).float()
        dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [T, T]
        # Soft attention weights to compute "effective" distance like AKT does.
        # Approximation: exponential decay by gamma * dist
        gamma = F.softplus(self.gamma).view(1, self.n_heads, 1, 1)
        decay = -gamma * dist.unsqueeze(0).unsqueeze(0)  # [1, H, T, T]
        scores = scores + decay

        # Causal mask
        scores = scores.masked_fill(causal_mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.d_k)
        return self.out(out)


class AKTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = MonotonicAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        h = self.norm1(x + self.dropout(self.attn(x, x, x, causal_mask)))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


class AKTModel(nn.Module):
    """AKT: Rasch-style embeddings (skill + difficulty) + monotonic attention.

    For our setup we use skill_id as the question identifier (no question-text
    features). This is the standard AKT-NCAT variant and is what most papers
    benchmark against.
    """
    def __init__(self, n_skills, d_model=128, n_heads=4, n_blocks=2, dropout=0.2):
        super().__init__()
        self.n_skills = n_skills
        # Question (skill) embedding
        self.q_emb = nn.Embedding(n_skills + 1, d_model, padding_idx=0)
        # Question difficulty (Rasch-style scalar)
        self.q_diff = nn.Embedding(n_skills + 1, 1, padding_idx=0)
        # Per-question variation embedding (separate from base) — proper AKT Rasch
        self.q_var = nn.Embedding(n_skills + 1, d_model, padding_idx=0)
        # Knowledge-acquisition base embedding (skill, correct) pair
        self.qa_emb = nn.Embedding(2 * n_skills + 1, d_model, padding_idx=0)
        # qa variation embedding
        self.qa_var = nn.Embedding(2 * n_skills + 1, d_model, padding_idx=0)
        # Position embedding
        self.pos_emb = nn.Embedding(512, d_model)

        self.q_blocks = nn.ModuleList([
            AKTBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])
        self.qa_blocks = nn.ModuleList([
            AKTBlock(d_model, n_heads, dropout) for _ in range(n_blocks)
        ])
        # Cross-attention from question history to knowledge state
        self.cross = MonotonicAttention(d_model, n_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

        self.out = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_inter, x_skill):
        """
        x_inter: [B, T] encoded interaction = skill + correct * n_skills
        x_skill: [B, T] target skill ids
        """
        B, T = x_inter.shape
        pos = torch.arange(T, device=x_inter.device).unsqueeze(0)

        # Rasch embedding: base + difficulty * variation (separate embeddings)
        diff = self.q_diff(x_skill + 1)                                  # [B, T, 1]
        q_e = self.q_emb(x_skill + 1) + diff * self.q_var(x_skill + 1)
        q_e = self.dropout(q_e + self.pos_emb(pos))

        qa_e = self.qa_emb(x_inter + 1) + diff * self.qa_var(x_inter + 1)
        qa_e = self.dropout(qa_e + self.pos_emb(pos))

        # Causal mask
        causal = torch.tril(torch.ones(T, T, device=x_inter.device))
        causal = causal.unsqueeze(0).unsqueeze(0)

        # Encode question and qa streams
        for block in self.q_blocks:
            q_e = block(q_e, causal)
        for block in self.qa_blocks:
            qa_e = block(qa_e, causal)

        # Cross-attention: query=questions, key/value=qa
        h = self.cross(q_e, qa_e, qa_e, causal)
        h = self.norm(h + q_e)

        # Concat current question encoding + cross-attended state
        feat = torch.cat([h, q_e], dim=-1)
        return torch.sigmoid(self.out(feat)).squeeze(-1)


class AKT:
    def __init__(self, n_skills, d_model=128, n_heads=4, n_blocks=2,
                 lr=1e-3, batch_size=64, epochs=10, device=None,
                 weight_decay=1e-5):
        self.n_skills = n_skills
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = AKTModel(n_skills, d_model, n_heads, n_blocks=n_blocks).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        self.criterion = nn.BCELoss(reduction='none')

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        total_loss, n_batches = 0, 0
        all_preds, all_labels = [], []
        ctx = torch.no_grad() if not train else torch.enable_grad()

        with ctx:
            for x_pad, y_pad, c_pad, mask in loader:
                x_pad = x_pad.to(self.device)
                y_pad = y_pad.to(self.device)
                c_pad = c_pad.to(self.device)
                mask = mask.to(self.device)

                if x_pad.shape[1] < 2:
                    continue

                # Predict response on next item from history up to t
                x_hist = x_pad[:, :-1]
                x_query = y_pad[:, 1:]
                target = c_pad[:, 1:]
                m = mask[:, 1:]

                preds = self.model(x_hist, x_query)
                loss = (self.criterion(preds, target) * m).sum() / m.sum().clamp(min=1)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1

                for b in range(x_pad.shape[0]):
                    all_preds.extend(preds[b][m[b]].detach().cpu().numpy())
                    all_labels.extend(target[b][m[b]].cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)
        return avg_loss, np.array(all_preds), np.array(all_labels)

    def fit(self, train_df, val_df=None):
        train_ds = KTDataset(train_df, self.n_skills)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, collate_fn=collate_fn)
        best_val_auc, best_state = 0, None

        for epoch in range(self.epochs):
            loss, _, _ = self._run_epoch(train_loader, train=True)
            if val_df is not None:
                val_ds = KTDataset(val_df, self.n_skills)
                val_loader = DataLoader(val_ds, batch_size=self.batch_size,
                                        shuffle=False, collate_fn=collate_fn)
                _, preds, labels = self._run_epoch(val_loader, train=False)
                auc = roc_auc_score(labels, preds)
                acc = accuracy_score(labels, (preds > 0.5).astype(int))
                print(f"  Epoch {epoch+1}/{self.epochs} | loss={loss:.4f} "
                      f"| val AUC={auc:.4f} | val ACC={acc:.4f}")
                if auc > best_val_auc:
                    best_val_auc = auc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                print(f"  Epoch {epoch+1}/{self.epochs} | loss={loss:.4f}")

        if best_state:
            self.model.load_state_dict(best_state)

    def evaluate(self, test_df):
        test_ds = KTDataset(test_df, self.n_skills)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, collate_fn=collate_fn)
        _, preds, labels = self._run_epoch(test_loader, train=False)
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, (preds > 0.5).astype(int))
        return {'AUC': auc, 'ACC': acc, 'preds': preds, 'labels': labels}
