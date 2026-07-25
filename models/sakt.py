"""
Self-Attentive Knowledge Tracing (SAKT) — Pandey & Karypis 2019
Transformer-style self-attention over interaction sequences.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from models.dkt import KTDataset, collate_fn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        def split(x):
            return x.view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        Q, K, V = split(self.q(query)), split(self.k(key)), split(self.v(value))
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.d_k)
        return self.out(out)


class SAKTModel(nn.Module):
    def __init__(self, n_skills, d_model=128, n_heads=4, dropout=0.2, n_blocks=2):
        super().__init__()
        self.n_skills = n_skills
        # Interaction embedding: (skill, correct) pair
        self.inter_emb = nn.Embedding(2 * n_skills + 1, d_model, padding_idx=0)
        # Exercise embedding: just skill
        self.exer_emb  = nn.Embedding(n_skills + 1, d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(512, d_model)  # position embedding

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadAttention(d_model, n_heads, dropout),
                'ff':   nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
            })
            for _ in range(n_blocks)
        ])
        self.out = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, exer):
        # x:    [B, T] encoded interaction history (shifted right)
        # exer: [B, T] target exercise ids
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        inter = self.dropout(self.inter_emb(x + 1) + self.pos_emb(pos))
        exer_e = self.dropout(self.exer_emb(exer + 1))

        # Causal mask: each position can only attend to previous positions
        causal_mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        h = inter
        for block in self.blocks:
            attn_out = block['attn'](exer_e, h, h, causal_mask)
            h = block['norm1'](exer_e + self.dropout(attn_out))
            ff_out = block['ff'](h)
            h = block['norm2'](h + self.dropout(ff_out))

        return torch.sigmoid(self.out(h)).squeeze(-1)  # [B, T]


class SAKT:
    def __init__(self, n_skills, d_model=128, n_heads=4, n_blocks=2,
                 lr=1e-3, batch_size=64, epochs=10, device=None):
        self.n_skills = n_skills
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SAKTModel(n_skills, d_model, n_heads, n_blocks=n_blocks).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
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
                mask  = mask.to(self.device)

                if x_pad.shape[1] < 2:
                    continue

                # Input: history up to t-1, query: skill at t
                x_hist  = x_pad[:, :-1]       # [B, T-1]
                x_query = y_pad[:, 1:]         # [B, T-1]
                target  = c_pad[:, 1:]         # [B, T-1]
                m       = mask[:, 1:]          # [B, T-1]

                preds = self.model(x_hist, x_query)  # [B, T-1]
                loss = (self.criterion(preds, target) * m).sum() / m.sum()

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

                for b in range(x_pad.shape[0]):
                    valid = m[b].cpu().numpy().astype(bool)
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
