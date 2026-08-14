#   python norros_f.py f_sns100.csv 0.1 24

import argparse

import numpy as np
from scipy import integrate

from linearP_f import fgn_autocov


def g_T(a, t, T, H):
    """Norros prediction kernel g_T(a, -t). Returns 0 outside 0 < t < T."""
    if t <= 0 or t >= T:
        return 0.0

    def integrand(v):
        return (v ** (H - 0.5)) * ((v + T) ** (H - 0.5)) / (v + t)

    inner, _ = integrate.quad(integrand, 0, a, limit=200)
    coefficient = np.sin(np.pi * (H - 0.5)) / np.pi
    power_term = (t ** (-H + 0.5)) * ((T - t) ** (-H + 0.5))
    return coefficient * power_term * inner


_GX, _GW = np.polynomial.legendre.leggauss(400)


def _gl(f, lo, hi):
    """Fixed 400-node Gauss-Legendre. Deterministic and smooth, unlike adaptive
    quadrature, so it can safely be nested inside another rule."""
    mid, rad = 0.5 * (lo + hi), 0.5 * (hi - lo)
    return np.sum(_GW * f(mid + rad * _GX)) * rad


def _g_T_gl(a, t, T, H):
    """g_T with the inner v-integral by fixed Gauss-Legendre, vectorised in t.

    Substituting v = a u^2 turns v^(H-1/2) dv into a smooth integrand (the
    Jacobian 2au cancels the mild vanishing power), so a fixed rule is exact to
    machine precision here and, unlike integrate.quad, introduces no
    tolerance-level noise for the outer rule to trip over.
    """
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros_like(t)
    # A node can round to exactly 0 or T under the endpoint substitution, where
    # the powers below are infinite; g_T is 0 outside (0, T), so mask first
    # rather than computing and discarding.
    ok = (t > 0) & (t < T)
    if not ok.any():
        return out
    tk = t[ok]

    u = 0.5 + 0.5 * _GX
    v = a * u * u
    # (nodes_v, nodes_t): one inner quadrature per outer node, vectorised
    integrand = ((v ** (H - 0.5)) * ((v + T) ** (H - 0.5)) * 2 * a * u)[:, None] \
        / (v[:, None] + tk[None, :])
    inner = 0.5 * (_GW @ integrand)
    coefficient = np.sin(np.pi * (H - 0.5)) / np.pi
    out[ok] = coefficient * (tk ** (0.5 - H)) * ((T - tk) ** (0.5 - H)) * inner
    return out


def norros_weights(T, H, horizon=1.0, quad_limit=200, method="gl"):

    T = float(T)
    if method == "quad":
        return np.array([
            integrate.quad(lambda t: g_T(horizon, t, T, H), k, k + 1,
                           limit=quad_limit)[0]
            for k in range(int(T))
        ])
    if method != "gl":
        raise ValueError(f"unknown method {method!r}")

    p = 2.0 / (1.5 - H)
    w = np.empty(int(T))
    for k in range(int(T)):
        if k == 0:                                   # singular at t = 0
            w[k] = _gl(lambda u: _g_T_gl(horizon, u ** p, T, H)
                       * p * u ** (p - 1), 0.0, 1.0)
        elif k == int(T) - 1:                        # singular at t = T
            w[k] = _gl(lambda u: _g_T_gl(horizon, T - u ** p, T, H)
                       * p * u ** (p - 1), 0.0, 1.0)
        else:
            w[k] = _gl(lambda t: _g_T_gl(horizon, t, T, H), float(k),
                       float(k) + 1.0)
    return w


def norros_step_weights(T, H, steps=1, **kw):

    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    cum = [norros_weights(T, H, horizon=float(j), **kw)
           for j in range(1, steps + 1)]
    return np.stack([cum[0]] + [cum[j] - cum[j - 1] for j in range(1, steps)])


def design(Z, T, horizon=1):
    """Rolling design matrix and cumulative target, shared by every scorer.

    Row i of `win` holds the T samples ending at Z[i + T - 1], most recent
    first; truth[i] is sum(Z[i+T : i+T+horizon]). Both have
    len(Z) - T - horizon + 1 rows.
    """
    win = np.lib.stride_tricks.sliding_window_view(Z, T)[:-horizon][:, ::-1]
    cs = np.concatenate(([0.0], np.cumsum(Z)))
    return win, cs[T + horizon:] - cs[T:len(Z) - horizon + 1]


def forecast(Z_test, w):
    """Rolling one-step forecasts. Returns (predictions, truth)."""
    T = len(w)
    windows = np.lib.stride_tricks.sliding_window_view(Z_test, T)[:-1][:, ::-1]
    return windows @ w, Z_test[T:]


def forecast_multistep(Z_test, W):

    steps, T = W.shape
    n = len(Z_test) - T - steps + 1
    if n < 1:
        raise ValueError(
            f"need > {T + steps - 1} test samples for T={T}, steps={steps}; "
            f"got {len(Z_test)}")
    windows = np.lib.stride_tricks.sliding_window_view(Z_test, T)[:n, ::-1]
    truth = np.stack([Z_test[T + j:T + j + n] for j in range(steps)], axis=1)
    return windows @ W.T, truth


def nmse(pred, truth):
    return np.mean((truth - pred) ** 2) / np.var(truth)


def main():
    import pandas as pd
    from analyze_traffic import analyze_traffic   # variance-time H

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="f_sns100.csv")
    ap.add_argument("dt", nargs="?", type=float, default=0.1)
    ap.add_argument("T", nargs="?", type=int, default=24)
    ap.add_argument("--steps", type=int, default=1,
                    help="forecast Z[t+1] .. Z[t+steps] individually, by "
                         "differencing the kernel in a; scored per step")
    args = ap.parse_args()

    df = pd.read_csv(f"./dataset/{args.path}", skiprows=1).dropna().reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(n * 0.7), 1].values
    test = df.iloc[int(n * 0.8):, 1].values

    m, a, H = analyze_traffic(train, args.dt)
    Z_test = (test - m * args.dt) / np.sqrt(m * a)

    if args.steps > 1:
        W = norros_step_weights(args.T, H, args.steps)
        pred, truth = forecast_multistep(Z_test, W)
        print(f"{args.path}  T={args.T}  H_est={H:.4f}  steps={args.steps}")
        for j in range(args.steps):
            print(f"  step {j + 1:>2}-ahead  NMSE = {nmse(pred[:, j], truth[:, j]):.4f}"
                  f"   (sum w = {W[j].sum():.4f})")
        print(f"  pooled       NMSE = {nmse(pred.ravel(), truth.ravel()):.4f}"
              f"   over {len(pred)} origins")
        print(f"  step-1 first 6 weights: {np.round(W[0, :6], 4)}")
    else:
        w = norros_weights(args.T, H)
        pred, truth = forecast(Z_test, w)
        print(f"{args.path}  T={args.T}  H_est={H:.4f}")
        print(f"  Norros g_T   NMSE = {nmse(pred, truth):.4f}   (sum w = {w.sum():.4f})")
        print(f"  first 6 weights: {np.round(w[:6], 4)}")


if __name__ == "__main__":
    main()
