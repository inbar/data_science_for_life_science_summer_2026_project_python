"""Nonlinear + conditional cell of the design: an MLP classifier and its
Integrated-Gradients attributions.

Architecture decision
----------------------
Per the brief the network uses ``tanh`` hidden activations and a ``softmax``
output, so predictions live on the probability simplex. It is trained with
cross-entropy on the *same* shared rank-transformed/standardised HVG matrix the
other three methods see, so the comparison stays fair. Early stopping on a held
-out validation loss prevents the small net from over-fitting.

Integrated Gradients (Sundararajan et al., 2017) then attributes each class
prediction back to the input genes. We average the per-cell attributions over
the cells *of* the target class, giving a signed gene x class attribution matrix
(positive = the gene pushes the model toward that class).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients
from sklearn.model_selection import train_test_split

# TODO: refactor

# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class TanhSoftmaxMLP(nn.Module):
    """Small MLP: Linear -> tanh (x depth) -> Linear -> (softmax via CE loss)."""

    def __init__(self, n_in: int, n_classes: int, hidden=(256, 128), dropout=0.2):
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # returns logits; softmax applied in loss / attribution
        return self.net(x)

    def embed(self, x):
        """Penultimate-layer representation (for the MLP-map UMAP)."""
        h = x
        for layer in list(self.net)[:-1]:
            h = layer(h)
        return h


@dataclass
class TrainHistory:
    train_loss: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    best_epoch: int = 0


@dataclass
class TrainedMLP:
    model: TanhSoftmaxMLP
    history: TrainHistory
    classes: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    test_acc: float
    test_macro_f1: float


def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_core(X, y_idx, n_classes, hidden, dropout, lr, weight_decay,
              max_epochs, patience, batch_size, val_size, seed, device,
              verbose=False):
    """Core trainer: stratified internal train/val split of (X, y_idx), Adam +
    inverse-frequency class weights, early stopping on validation loss. Returns
    (best_model, history). Shared by ``train_mlp`` and cross-validation."""
    idx = np.arange(len(y_idx))
    idx_tr, idx_val = train_test_split(idx, test_size=val_size,
                                       stratify=y_idx, random_state=seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y_idx, dtype=torch.long)
    counts = np.bincount(y_idx[idx_tr], minlength=n_classes).astype(float)
    w = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)),
                     dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=w)
    model = TanhSoftmaxMLP(X.shape[1], n_classes, hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    Xtr, ytr = Xt[idx_tr].to(device), yt[idx_tr].to(device)
    Xva, yva = Xt[idx_val].to(device), yt[idx_val].to(device)
    hist = TrainHistory()
    best_val, best_state, wait = np.inf, None, 0
    n_tr = len(idx_tr)
    rng = np.random.default_rng(seed)
    for epoch in range(max_epochs):
        model.train()
        perm = rng.permutation(n_tr)
        running = 0.0
        for s in range(0, n_tr, batch_size):
            b = perm[s : s + batch_size]
            opt.zero_grad()
            loss = criterion(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            running += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            logits = model(Xva)
            vloss = criterion(logits, yva).item()
            vacc = (logits.argmax(1) == yva).float().mean().item()
        hist.train_loss.append(running / n_tr)
        hist.val_loss.append(vloss)
        hist.val_acc.append(vacc)
        if verbose:
            print(f"epoch {epoch:3d}  train {hist.train_loss[-1]:.4f}  "
                  f"val {vloss:.4f}  acc {vacc:.3f}")
        if vloss < best_val - 1e-4:
            best_val, wait, hist.best_epoch = vloss, 0, epoch
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, hist, idx_val


def train_mlp(
    X: np.ndarray,
    y: np.ndarray,
    hidden=(512, 256),          # tuned by CV macro-F1 sweep (run_pipeline.py tune)
    dropout: float = 0.4,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 256,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 0,
    device: str | None = None,
    verbose: bool = False,
) -> TrainedMLP:
    """Train the classifier with stratified train/val/test split + early stopping.

    Class imbalance is handled with inverse-frequency weights in the loss.
    """
    from sklearn.metrics import f1_score
    _set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    classes, y_idx = np.unique(y, return_inverse=True)
    n_classes = len(classes)

    idx = np.arange(len(y))
    idx_fit, idx_te = train_test_split(
        idx, test_size=test_size, stratify=y_idx, random_state=seed
    )
    val_rel = val_size / (1.0 - test_size)   # val as a fraction of the fit set
    model, hist, _ = _fit_core(
        X[idx_fit], y_idx[idx_fit], n_classes, hidden, dropout, lr, weight_decay,
        max_epochs, patience, batch_size, val_rel, seed, device, verbose)

    with torch.no_grad():
        pred_te = model(torch.tensor(X[idx_te], dtype=torch.float32, device=device)
                        ).argmax(1).cpu().numpy()
    test_acc = float((pred_te == y_idx[idx_te]).mean())
    test_macro_f1 = float(f1_score(y_idx[idx_te], pred_te, average="macro"))

    return TrainedMLP(model=model, history=hist, classes=classes,
                      val_idx=np.array([], dtype=int), test_idx=idx_te,
                      test_acc=test_acc, test_macro_f1=test_macro_f1)


# --------------------------------------------------------------------------- #
# Integrated Gradients
# --------------------------------------------------------------------------- #
def integrated_gradients(
    trained: TrainedMLP,
    X: np.ndarray,
    y: np.ndarray,
    baseline: str = "median",
    n_steps: int = 50,
    max_cells_per_class: int = 1500,
    seed: int = 0,
    device: str | None = None,
) -> np.ndarray:
    """Signed IG attributions, returned as a (genes x classes) matrix.

    For each class the attribution is averaged over (a subsample of) the cells
    that truly belong to that class, attributing the model's logit for that
    class. ``baseline`` is the feature-wise median ("median") or zeros ("zero").
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = trained.model.to(device).eval()
    classes = trained.classes
    _, y_idx = np.unique(y, return_inverse=True)

    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    if baseline == "median":
        base = torch.median(Xt, dim=0, keepdim=True).values
    else:
        base = torch.zeros((1, X.shape[1]), device=device)

    ig = IntegratedGradients(model)
    rng = np.random.default_rng(seed)
    out = np.zeros((X.shape[1], len(classes)), dtype=float)

    for k in range(len(classes)):
        cells = np.where(y_idx == k)[0]
        if len(cells) == 0:
            continue
        if len(cells) > max_cells_per_class:
            cells = rng.choice(cells, max_cells_per_class, replace=False)
        inp = Xt[cells].clone().requires_grad_(True)
        attr = ig.attribute(inp, baselines=base, target=int(k), n_steps=n_steps)
        out[:, k] = attr.detach().cpu().numpy().mean(axis=0)
    return out


def mlp_embedding(trained: TrainedMLP, X: np.ndarray, device: str | None = None):
    """Penultimate-layer embedding for every cell (for the MLP-map UMAP)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = trained.model.to(device).eval()
    with torch.no_grad():
        emb = model.embed(torch.tensor(X, dtype=torch.float32, device=device))
    return emb.cpu().numpy()


# --------------------------------------------------------------------------- #
# Hyperparameter search & per-class evaluation
# --------------------------------------------------------------------------- #
def cross_validate(X, y, hidden=(256, 128), dropout=0.2, lr=1e-3,
                   weight_decay=1e-5, max_epochs=200, patience=15,
                   batch_size=256, inner_val=0.15, n_splits=3, seed=0,
                   device=None):
    """Stratified k-fold CV. Returns mean/std accuracy & macro-F1 and the pooled
    out-of-fold (y_true, y_pred). Macro-F1 weights every cell type equally, so it
    rewards getting the *minority* lineages right, not just the abundant ones."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes, y_idx = np.unique(y, return_inverse=True)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, f1s, yt_all, yp_all = [], [], [], []
    for tr, te in skf.split(X, y_idx):
        _set_seed(seed)
        model, _, _ = _fit_core(
            X[tr], y_idx[tr], len(classes), hidden, dropout, lr, weight_decay,
            max_epochs, patience, batch_size, inner_val, seed, device)
        with torch.no_grad():
            pred = model(torch.tensor(X[te], dtype=torch.float32, device=device)
                         ).argmax(1).cpu().numpy()
        accs.append(accuracy_score(y_idx[te], pred))
        f1s.append(f1_score(y_idx[te], pred, average="macro"))
        yt_all.append(classes[y_idx[te]]); yp_all.append(classes[pred])
    return {"acc": float(np.mean(accs)), "acc_std": float(np.std(accs)),
            "macro_f1": float(np.mean(f1s)), "macro_f1_std": float(np.std(f1s)),
            "y_true": np.concatenate(yt_all), "y_pred": np.concatenate(yp_all)}


def hyperparameter_search(X, y, grid, n_splits=3, seed=0, verbose=True):
    """Grid search over a list of config dicts; ranks by CV macro-F1.

    Each config may set hidden/dropout/lr/weight_decay (architecture stays
    tanh+softmax). Returns a DataFrame sorted best-first."""
    import pandas as pd
    rows = []
    for i, cfg in enumerate(grid):
        r = cross_validate(X, y, n_splits=n_splits, seed=seed, **cfg)
        rows.append({**cfg, "cv_acc": r["acc"], "cv_acc_std": r["acc_std"],
                     "cv_macro_f1": r["macro_f1"],
                     "cv_macro_f1_std": r["macro_f1_std"]})
        if verbose:
            print(f"[{i+1}/{len(grid)}] {cfg} -> acc {r['acc']:.3f} "
                  f"macroF1 {r['macro_f1']:.3f}", flush=True)
    return pd.DataFrame(rows).sort_values("cv_macro_f1",
                                          ascending=False).reset_index(drop=True)


def per_class_f1(y_true, y_pred):
    """Per-class F1 as a pandas Series indexed by class label."""
    import pandas as pd
    from sklearn.metrics import f1_score
    labels = np.unique(np.concatenate([y_true, y_pred]))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return pd.Series(f1, index=labels, name="f1").sort_values()
