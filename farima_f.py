#   python farima_f.py f_sns100.csv 0.1 24 --p 2
#   python farima_f.py f_sns100.csv 0.1 24 --p 2 --steps 3

import argparse

import numpy as np
import scipy.linalg
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri

_R_READY = False


def _ensure_r():
    global _R_READY
    if not _R_READY:
        numpy2ri.activate()
        ro.r('suppressMessages(library(arfima))')
        _R_READY = True


def fit_arfima(Z_train, p=2):
    """Fit ARFIMA(p, d, 0) once on the training series. Returns (d, phi)."""
    _ensure_r()
    ro.globalenv['y'] = np.asarray(Z_train, dtype=float)
    ro.globalenv['p'] = int(p)
    ro.r('fit <- arfima(y, order = c(p, 0, 0), quiet = TRUE)')
    ro.r('mode1 <- fit$modes[[1]]')
    d = float(np.asarray(ro.r('mode1$dfrac'))[0])
    phi = np.asarray(
        ro.r('if (length(mode1$phi)) mode1$phi else numeric(0)'), dtype=float)
    return d, phi


def _model_acvf(maxlag):
    """Theoretical autocovariance of the model left in the R global `mode1`."""
    return np.asarray(
        ro.r(f'tacvfARFIMA(phi = if (length(mode1$phi)) mode1$phi else numeric(0),'
             f' dfrac = mode1$dfrac, maxlag = {int(maxlag)})'), dtype=float)


def arfima_weights(Z_train, T, p=2, route="acvf", ridge=1e-8, horizon=1):
    """Fit once, then return the T-tap filter implied by the model.

    `horizon` = h predicts the *cumulative* demand over the next h cells,
    sum_{j=1..h} Z[t+j], matching the target of the Norros g_T kernel with
    a = h (see norros_f.norros_weights). h = 1 is the plain one-step filter.

    The optimal linear predictor of a sum is the sum of the optimal predictors,
    so the Toeplitz system keeps the same Sigma and its right-hand side becomes
    sum_{j=1..h} gamma(j .. j+T-1).

    w[0] multiplies the most recent increment, so apply it to a reversed window.
    Returns (w, d, phi).
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    d, phi = fit_arfima(Z_train, p)

    if route == "acvf":
        acvf = _model_acvf(T + horizon)
        Sigma = scipy.linalg.toeplitz(acvf[:T])
        rhs = sum(acvf[j:T + j] for j in range(1, horizon + 1))
        w = np.linalg.solve(Sigma + ridge * np.eye(T), rhs)
    elif route == "arinf":
        # The truncated AR(inf) expansion is inherently one-step; an h-step
        # version requires iterating the filter, which the acvf route already
        # does exactly. Kept as a h=1 cross-check only.
        if horizon != 1:
            raise NotImplementedError(
                "route='arinf' supports horizon=1 only; use route='acvf'")
        c = np.empty(T + 1)                    # coefficients of (1-B)^d
        c[0] = 1.0
        for j in range(1, T + 1):
            c[j] = c[j - 1] * (j - 1 - d) / j
        pi = np.convolve(np.concatenate(([1.0], -phi)), c)[:T + 1]
        w = -pi[1:T + 1]
    else:
        raise ValueError(f"unknown route {route!r}")
    return w, d, phi


def arfima_step_weights(Z_train, T, p=2, steps=1, ridge=1e-8):
    """Fit once, then return one T-tap filter *per step* of the trajectory.

    Where `arfima_weights(horizon=h)` collapses the next h cells into a single
    cumulative target, this keeps them separate: row j of the returned W is the
    direct predictor of Z[t+1+j], for j = 0 .. steps-1. That is farima.py's
    `n.ahead` semantics (h forecasts, scored individually) but with the model
    estimated a single time instead of refit on every window.

    These are *direct* predictors -- each step gets its own Toeplitz solve
    against gamma(j+1 .. j+T) -- not the one-step filter iterated forward. For a
    correctly specified model the two coincide; direct is used here because it
    stays exact when the model is misspecified, and Sigma is factored once so
    all `steps` solves cost little more than one.

    Row 0 equals `arfima_weights(..., horizon=1)[0]`. As with that function,
    W[:, 0] multiplies the most recent increment, so apply to reversed windows.
    Returns (W, d, phi) with W of shape (steps, T).
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    d, phi = fit_arfima(Z_train, p)

    acvf = _model_acvf(T + steps)
    Sigma = scipy.linalg.toeplitz(acvf[:T]) + ridge * np.eye(T)
    # One factorization reused across steps; only the right-hand side moves.
    cho = scipy.linalg.cho_factor(Sigma)
    W = np.stack([scipy.linalg.cho_solve(cho, acvf[j:T + j])
                  for j in range(1, steps + 1)])
    return W, d, phi


def main():
    import pandas as pd
    from analyze_traffic import analyze_traffic     # variance-time H
    from norros_f import forecast, forecast_multistep, nmse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="f_sns100.csv")
    ap.add_argument("dt", nargs="?", type=float, default=0.1)
    ap.add_argument("T", nargs="?", type=int, default=24)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--route", choices=["acvf", "arinf"], default="acvf")
    ap.add_argument("--steps", type=int, default=1,
                    help="forecast Z[t+1] .. Z[t+steps] individually "
                         "(farima.py's n.ahead); scored per step")
    args = ap.parse_args()

    df = pd.read_csv(f"./dataset/{args.path}", skiprows=1).dropna().reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(n * 0.7), 1].values
    test = df.iloc[int(n * 0.8):, 1].values

    m, a, H = analyze_traffic(train, args.dt)
    Z_train = (train - m * args.dt) / np.sqrt(m * a)
    Z_test = (test - m * args.dt) / np.sqrt(m * a)

    if args.steps > 1:
        if args.route != "acvf":
            ap.error("--steps > 1 requires --route acvf")
        W, d, phi = arfima_step_weights(Z_train, args.T, args.p, args.steps)
        pred, truth = forecast_multistep(Z_test, W)
        print(f"{args.path}  T={args.T}  H_est={H:.4f}  steps={args.steps}")
        print(f"  fitted ARFIMA({args.p},d,0): d = {d:.4f}  phi = {np.round(phi, 4)}")
        for j in range(args.steps):
            print(f"  step {j + 1:>2}-ahead  NMSE = {nmse(pred[:, j], truth[:, j]):.4f}"
                  f"   (sum w = {W[j].sum():.4f})")
        print(f"  pooled       NMSE = {nmse(pred.ravel(), truth.ravel()):.4f}"
              f"   over {len(pred)} origins")
        print(f"  step-1 first 6 weights: {np.round(W[0, :6], 4)}")
    else:
        w, d, phi = arfima_weights(Z_train, args.T, args.p, args.route)
        pred, truth = forecast(Z_test, w)
        print(f"{args.path}  T={args.T}  H_est={H:.4f}")
        print(f"  fitted ARFIMA({args.p},d,0): d = {d:.4f}  phi = {np.round(phi, 4)}")
        print(f"  ARFIMA       NMSE = {nmse(pred, truth):.4f}   (sum w = {w.sum():.4f})")
        print(f"  first 6 weights: {np.round(w[:6], 4)}")


if __name__ == "__main__":
    main()
