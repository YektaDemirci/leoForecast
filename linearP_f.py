#   python linearP_f.py f_sns100.csv 0.1 24

import argparse

import numpy as np
from scipy.linalg import toeplitz


def fgn_autocov(k, H):
    """Autocovariance of unit-variance fractional Gaussian noise at lag k."""
    k = np.abs(np.asarray(k, dtype=float))
    return 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * k ** (2 * H)
                  + (k + 1) ** (2 * H))


def linearp_weights(T, H, horizon=1, ridge=1e-10):

    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    g = fgn_autocov(np.arange(T + horizon + 1), H)
    rhs = sum(g[j:T + j] for j in range(1, horizon + 1))
    return np.linalg.solve(toeplitz(g[:T]) + ridge * np.eye(T), rhs)


def main():
    import pandas as pd
    from analyze_traffic import analyze_traffic     # variance-time H
    from norros_f import design, norros_weights           # deferred: avoids a cycle

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", nargs="?", default="f_sns100.csv")
    ap.add_argument("dt", nargs="?", type=float, default=0.1)
    ap.add_argument("T", nargs="?", type=int, default=24)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()

    df = pd.read_csv(f"./dataset/{args.path}", skiprows=1).dropna().reset_index(drop=True)
    n = len(df)
    train = df.iloc[:int(n * 0.7), 1].values
    test = df.iloc[int(n * 0.8):, 1].values

    m, a, H = analyze_traffic(train, args.dt)
    Z_test = (test - m * args.dt) / np.sqrt(m * a)

    win, truth = design(Z_test, args.T, args.horizon)
    denom = np.var(truth)

    w_lp = linearp_weights(args.T, H, args.horizon)
    w_gt = norros_weights(args.T, H, horizon=float(args.horizon))
    e_lp = np.mean((truth - win @ w_lp) ** 2) / denom
    e_gt = np.mean((truth - win @ w_gt) ** 2) / denom

    print(f"{args.path}  T={args.T}  h={args.horizon}  H_est={H:.4f}")
    print(f"  linearP      NMSE = {e_lp:.4f}   (sum w = {w_lp.sum():.4f})")
    print(f"  Norros g_T   NMSE = {e_gt:.4f}   (sum w = {w_gt.sum():.4f})")
    print(f"  g_T discretization penalty: {100*(e_gt-e_lp)/e_lp:+.2f}%")
    print(f"  first 6 weights: {np.round(w_lp[:6], 4)}")


if __name__ == "__main__":
    main()
