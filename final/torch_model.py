"""PyTorch residual MLP for tabular data, wrapped as a scikit-learn estimator.
Differences from the sklearn MLP of the progress report: residual blocks + LayerNorm, SiLU, dropout,
AdamW with weight decay, one-cycle learning-rate schedule, a fixed epoch budget (no early stopping on
validation data) and an ensemble of several random seeds."""
import numpy as np, torch, torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

torch.set_num_threads(2)

class Block(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.SiLU(), nn.Dropout(p), nn.Linear(d, d))
    def forward(self, x):
        return x + self.f(x)

class Net(nn.Module):
    def __init__(self, d_in, width, n_blocks, p):
        super().__init__()
        self.inp = nn.Linear(d_in, width)
        self.blocks = nn.Sequential(*[Block(width, p) for _ in range(n_blocks)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.SiLU(), nn.Linear(width, 1))
    def forward(self, x):
        return self.out(self.blocks(self.inp(x))).squeeze(-1)

class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, width=128, n_blocks=2, dropout=0.1, lr=3e-3, weight_decay=1e-3,
                 epochs=60, batch_size=128, n_seeds=3, random_state=42):
        self.width, self.n_blocks, self.dropout, self.lr = width, n_blocks, dropout, lr
        self.weight_decay, self.epochs, self.batch_size = weight_decay, epochs, batch_size
        self.n_seeds, self.random_state = n_seeds, random_state

    def fit(self, X, y, sample_weight=None):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        yt = torch.tensor(np.asarray(y), dtype=torch.float32)
        w = torch.ones(len(yt)) if sample_weight is None else torch.tensor(np.asarray(sample_weight), dtype=torch.float32)
        self.classes_ = np.array([0, 1])
        self.nets_ = []
        n = len(yt); steps = int(np.ceil(n / self.batch_size)) * self.epochs
        for s in range(self.n_seeds):
            g = torch.Generator().manual_seed(self.random_state + s)
            torch.manual_seed(self.random_state + s)
            net = Net(X.shape[1], self.width, self.n_blocks, self.dropout)
            opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)
            net.train()
            for _ in range(self.epochs):
                perm = torch.randperm(n, generator=g)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    if len(idx) < 2: continue
                    loss = (nn.functional.binary_cross_entropy_with_logits(net(X[idx]), yt[idx], reduction="none") * w[idx]).mean()
                    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            net.eval(); self.nets_.append(net)
        return self

    def predict_proba(self, X):
        X = torch.tensor(np.asarray(X), dtype=torch.float32)
        with torch.no_grad():
            p = np.mean([torch.sigmoid(n(X)).numpy() for n in self.nets_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def n_parameters(self):
        return sum(p.numel() for p in self.nets_[0].parameters()) * len(self.nets_)
