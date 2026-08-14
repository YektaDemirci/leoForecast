import os
import sys

import numpy as np
import torch
import torch.nn as nn

# The DLinear implementation lives in the repo root, next to this study.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.DLinear import Model as _DLinear


class _Cfg:
    """The attribute bag models.DLinear.Model reads its shape from.

    It only ever touches seq_len, pred_len, individual and enc_in, so the full
    run_longExp argparse namespace is not needed -- and constructing one here
    would drag in the whole Exp_Main data pipeline for a single-channel series
    that is already in memory as an array.
    """

    def __init__(self, seq_len, pred_len, enc_in=1, individual=False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.individual = individual


def _windows(Z, T, horizon=1):
    """Sliding design: X[k] = Z[k : k+T], y[k] = sum of the next `horizon`.

    Same target convention as linearp_weights and norros_weights -- the
    CUMULATIVE demand over the next `horizon` samples -- so the two planners are
    scored on the identical quantity. At horizon=1 that is just Z[k+T].

    Chronological order, NOT reversed: linearP's w[0] multiplies the most recent
    increment and so is applied to a flipped window, but DLinear's moving-average
    decomposition is direction-sensitive and expects time to run forwards.
    """
    n = len(Z) - T - horizon + 1
    if n < 1:
        raise ValueError(f"need > T + horizon = {T + horizon} samples to fit "
                         f"DLinear, got {len(Z)}")
    idx = np.arange(T)[None, :] + np.arange(n)[:, None]
    X = Z[idx]
    y = np.stack([Z[T + h: T + h + n] for h in range(horizon)]).sum(axis=0)
    return X.astype(np.float32), y.astype(np.float32)


def train_dlinear(s, T, horizon=1, epochs=60, batch_size=64, lr=1e-3,
                  val_frac=0.2, patience=8, seed=0, device=None, verbose=False):

    device = device or torch.device("cpu")
    s = np.asarray(s, dtype=float)

    # Fit the scaler on the training portion alone, before windowing, so the
    # held-out tail contributes nothing to mu/sd.
    n_fit = len(s) - max(1, int(round((len(s) - T - horizon + 1) * val_frac)))
    mu = float(s[:n_fit].mean())
    sd = float(s[:n_fit].std())
    if not sd > 0:
        raise ValueError("training series is constant; cannot standardize")
    X, y = _windows((s - mu) / sd, T, horizon)

    n_val = max(1, int(round(len(X) * val_frac)))
    n_tr = len(X) - n_val
    if n_tr < 1:
        raise ValueError(f"{len(X)} windows is too few to hold out {n_val} "
                         f"for validation; raise --train-samples")

    Xt = torch.from_numpy(X[:n_tr]).unsqueeze(-1).to(device)   # [N, T, 1]
    yt = torch.from_numpy(y[:n_tr]).to(device)                 # [N] cumulative
    Xv = torch.from_numpy(X[n_tr:]).unsqueeze(-1).to(device)
    yv = torch.from_numpy(y[n_tr:]).to(device)

    def cum(x):
        return net(x).squeeze(-1).sum(-1)

    torch.manual_seed(seed)
    net = _DLinear(_Cfg(T, horizon)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.MSELoss()

    best, best_state, bad = float("inf"), None, 0
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n_tr, device=device)
        for b in range(0, n_tr, batch_size):
            j = perm[b: b + batch_size]
            opt.zero_grad()
            loss = lossf(cum(Xt[j]), yt[j])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = lossf(cum(Xv), yv).item()
        if verbose:
            print(f"        epoch {ep:3d}  val_mse={v:.5f}")
        if v < best - 1e-6:
            best, bad = v, 0
            best_state = {k: t.detach().clone()
                          for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    # val NMSE, comparable to the numbers select_H reports for linearP. Scale
    # cancels in the ratio, so standardized units are fine here.
    return net, mu, sd, best / max(float(np.var(y[n_tr:])), 1e-12)


@torch.no_grad()
def predict_batch(net, windows, device=None):
    """Forecast a stack of standardized windows in one forward pass.

    `windows` is [N, T] in chronological order; returns [N], the CUMULATIVE
    demand over the net's pred_len steps -- the same quantity train_dlinear
    optimizes and the same one linearp_weights targets, so a forecast from
    either planner means the same thing.

    The whole test window is planned in a single call rather than one call per
    period: the forecast depends only on the history and the frozen scaler,
    both known in advance, so batching changes nothing about the result and
    takes the per-period python/torch dispatch out of the tick loop.
    """
    if device is None:
        device = next(net.parameters(), torch.zeros(1)).device
    x = torch.as_tensor(np.asarray(windows, dtype=np.float32),
                        device=device).unsqueeze(-1)
    return net(x).squeeze(-1).sum(-1).cpu().numpy()
