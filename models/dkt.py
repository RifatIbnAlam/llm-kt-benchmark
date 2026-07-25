"""
Deep Knowledge Tracing (DKT) — Piech et al. 2015
LSTM-based model. Input: one-hot (skill_id, correct) pairs.
Predicts P(correct on next question for each skill).
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score


class KTDataset(Dataset):
    def __init__(self, df, n_skills, max_seq=200, mode='last'):
        """
        mode:
          'last'   - keep only the last max_seq interactions per student
                     (default; matches pyKT and most KT papers)
          'window' - split long sequences into non-overlapping windows of
                     max_seq each (more training signal, but ~2x cost)
        """
        self.n_skills = n_skills
        self.max_seq = max_seq
        self.sequences = []
        for uid, group in df.groupby('user_id'):
            group = group.sort_values('order')
            skills = group['skill_id'].values.astype(int)
            corrects = group['correct'].values.astype(int)
            # Encode input: skill_id + correct -> index in [0, 2*n_skills)
            # 0..n_skills-1 = wrong, n_skills..2*n_skills-1 = correct
            x = skills + corrects * n_skills
            y = skills      # target: predict correctness on each skill
            c = corrects

            if mode == 'last' or len(x) <= max_seq:
                # Truncate to last max_seq (preserve most recent history)
                self.sequences.append((x[-max_seq:], y[-max_seq:], c[-max_seq:]))
            elif mode == 'window':
                for s in range(0, len(x), max_seq):
                    e = min(s + max_seq, len(x))
                    if e - s < 2:  # need at least 2 for next-step prediction
                        continue
                    self.sequences.append((x[s:e], y[s:e], c[s:e]))
            else:
                raise ValueError(f"Unknown mode: {mode}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x, y, c = self.sequences[idx]
        return x, y, c


def collate_fn(batch):
    """Pad sequences and create mask."""
    xs, ys, cs = zip(*batch)
    max_len = max(len(x) for x in xs)
    B = len(xs)
    x_pad = torch.zeros(B, max_len, dtype=torch.long)
    y_pad = torch.zeros(B, max_len, dtype=torch.long)
    c_pad = torch.zeros(B, max_len, dtype=torch.float)
    mask  = torch.zeros(B, max_len, dtype=torch.bool)
    for i, (x, y, c) in enumerate(zip(xs, ys, cs)):
        T = len(x)
        x_pad[i, :T] = torch.tensor(x)
        y_pad[i, :T] = torch.tensor(y)
        c_pad[i, :T] = torch.tensor(c, dtype=torch.float)
        mask[i, :T]  = True
    return x_pad, y_pad, c_pad, mask


class DKTModel(nn.Module):
    def __init__(self, n_skills, emb_dim=64, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.n_skills = n_skills
        self.emb = nn.Embedding(2 * n_skills + 1, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, n_skills)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T] — encoded (skill, correctness) indices
        e = self.dropout(self.emb(x + 1))  # +1 to reserve 0 for padding
        h, _ = self.lstm(e)
        logits = self.out(self.dropout(h))   # [B, T, n_skills]
        return torch.sigmoid(logits)


class DKT:
    def __init__(self, n_skills, emb_dim=64, hidden_dim=128,
                 lr=1e-3, batch_size=64, epochs=10, device=None):
        self.n_skills = n_skills
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DKTModel(n_skills, emb_dim, hidden_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss(reduction='none')

    def fit(self, train_df, val_df=None):
        dataset = KTDataset(train_df, self.n_skills)
        loader  = DataLoader(dataset, batch_size=self.batch_size,
                             shuffle=True, collate_fn=collate_fn)
        best_val_auc = 0
        best_state = None

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            n_batches = 0
            for x_pad, y_pad, c_pad, mask in loader:
                x_pad = x_pad.to(self.device)
                y_pad = y_pad.to(self.device)
                c_pad = c_pad.to(self.device)
                mask  = mask.to(self.device)

                preds = self.model(x_pad)  # [B, T, n_skills]

                # Shift: predict correctness at t+1 using output at t
                # preds[:, :-1, :] predicts c_pad[:, 1:]
                if x_pad.shape[1] < 2:
                    continue
                B, T, S = preds.shape
                # Gather predicted prob for the target skill at each step
                # target skill at t+1 is y_pad[:, 1:]
                pred_shift = preds[:, :-1, :]                    # [B, T-1, S]
                skill_idx  = y_pad[:, 1:].unsqueeze(-1)          # [B, T-1, 1]
                pred_skill = pred_shift.gather(2, skill_idx).squeeze(-1)  # [B, T-1]

                target = c_pad[:, 1:]                             # [B, T-1]
                m      = mask[:, 1:]                              # [B, T-1]

                loss = (self.criterion(pred_skill, target) * m).sum() / m.sum()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)

            if val_df is not None:
                metrics = self.evaluate(val_df)
                print(f"  Epoch {epoch+1}/{self.epochs} | loss={avg_loss:.4f} "
                      f"| val AUC={metrics['AUC']:.4f} | val ACC={metrics['ACC']:.4f}")
                if metrics['AUC'] > best_val_auc:
                    best_val_auc = metrics['AUC']
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                print(f"  Epoch {epoch+1}/{self.epochs} | loss={avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict(self, test_df):
        dataset = KTDataset(test_df, self.n_skills)
        loader  = DataLoader(dataset, batch_size=self.batch_size,
                             shuffle=False, collate_fn=collate_fn)
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for x_pad, y_pad, c_pad, mask in loader:
                x_pad = x_pad.to(self.device)
                preds = self.model(x_pad)

                B, T, S = preds.shape
                if T < 2:
                    continue
                pred_shift = preds[:, :-1, :]
                skill_idx  = y_pad[:, 1:].unsqueeze(-1)
                pred_skill = pred_shift.gather(2, skill_idx).squeeze(-1)
                target     = c_pad[:, 1:]
                m          = mask[:, 1:]

                for b in range(B):
                    valid = m[b].cpu().numpy().astype(bool)
                    all_preds.extend(pred_skill[b][m[b]].cpu().numpy())
                    all_labels.extend(target[b][m[b]].cpu().numpy())

        return np.array(all_preds), np.array(all_labels)

    def evaluate(self, test_df):
        preds, labels = self.predict(test_df)
        auc = roc_auc_score(labels, preds)
        acc = accuracy_score(labels, (preds > 0.5).astype(int))
        return {'AUC': auc, 'ACC': acc}
