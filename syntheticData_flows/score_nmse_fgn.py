"""The control experiment for score_nmse_traffic.py: the same three predictors,
the same split, the same scoring -- on EXACT fractional Gaussian noise.

score_nmse_traffic.py runs the comparison on aggregated ON/OFF traffic, where
farima_f beats linearP_f by 0.2-0.7% NMSE. That is either (a) misspecification
-- the ON/OFF aggregate is not fGn at finite source count and cell width, which
its acf_dev of 0.07-0.34 says loudly -- or (b) a bug in linearp_weights. The two
are indistinguishable on data whose true covariance we do not know.

Here we do know it. Davies-Harte circulant embedding draws from the fGn
covariance EXACTLY (not approximately: the method is a synthesis, and the
resulting sample has the fGn Toeplitz covariance to machine precision), so the
model linearp_weights assumes is the model that generated the data. Under that
premise linearP_f is the optimal linear predictor from T taps, and no
forecaster -- farima_f included -- can beat it by more than sampling noise.

So this run is a falsifiable test rather than another measurement:

  * --h oracle  hands the kernels the TRUE H. linearP_f is then exactly
                optimal and must win every seed. If it does not, the bug is in
                linearp_weights and the traffic result means nothing.
  * --h whittle estimates H the way score_nmse_traffic.py does. Any remaining
                gap is the price of estimating one parameter, measured on data
                with no misspecification left in it -- which is the number you
                need to subtract before reading the traffic gap as
                misspecification.

  vv/bin/python score_nmse_fgn.py --h oracle
  vv/bin/python score_nmse_fgn.py --h whittle

Defaults mirror nmseTrafficRuns.sh: T=48 taps, horizon 1, ARFIMA(1,d,0), 10
seeds, 9600 samples per realisation (the length of the ON/OFF traces).
"""

import argparse
import datetime
import os

import numpy as np
import pandas as pd
from scipy import stats

from single_cell_1s import run_one
from h_estimators import local_whittle
from score_nmse_traffic import ci

# The H values the traffic run covers, via H = (3 - alpha) / 2.
H_GRID = [(1.04, 0.98), (1.24, 0.88), (1.44, 0.78)]
DT = 0.15                # carried through only so run_one's units match
N_SAMPLES = 9600         # same length as the ON/OFF traces
METHODS = [("lp", "linearP_f"), ("far", "farima_f"),
           ("gt", "norros_f"), ("naive", "persist")]


def fgn_davies_harte(n, H, rng):
    """n samples of unit-variance fGn(H) by circulant embedding.

    Exact: the returned vector's covariance is the fGn Toeplitz covariance to
    machine precision, not an approximation of it. The circulant's first row is
    the autocovariance reflected about lag n; its eigenvalues are the real FFT
    of that row, and they are non-negative for fGn at every H in (0, 1), which
    is what makes the embedding valid here without the usual fallback.
    """
    k = np.arange(n + 1)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2.0 * k ** (2 * H)
               + (k + 1) ** (2 * H))
    row = np.concatenate([g, g[-2:0:-1]])          # length 2n
    lam = np.fft.fft(row).real
    if lam.min() < -1e-10 * lam.max():
        raise RuntimeError(f"circulant not PSD at H={H}: min eig {lam.min():g}")
    lam = np.clip(lam, 0.0, None)

    m = len(row)
    z = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    # The scaling that makes the real part of the transform have covariance g.
    x = np.fft.fft(np.sqrt(lam / (2.0 * m)) * z).real
    return x[:n]


def verify_covariance(H, rng, n=4096, reps=400):
    """Sanity check on the synthesiser itself, independent of any predictor.

    Averages the sample autocovariance over `reps` independent realisations and
    reports the worst deviation from the fGn target over lags 0..8. This is a
    check that fgn_davies_harte is correct; it is not a check of the data,
    which by construction cannot be wrong.

    Deliberately NOT demeaned. fGn has mean zero by construction, so the mean
    carries no information -- but subtracting the SAMPLE mean is not harmless
    at H near 1, where the sample mean is a poor estimate of zero and removing
    it strips out most of the low-frequency power the autocovariance is made
    of. That bias is why single_cell_1s.acov -- which does demean, correctly,
    since it runs on data with an unknown mean -- reports acf_dev of 0.32 even
    on the exact fGn drawn here. Demeaning would make this check measure that
    bias instead of the synthesiser.
    """
    acc = np.zeros(9)
    for _ in range(reps):
        x = fgn_davies_harte(n, H, rng)
        f = np.fft.rfft(x, 2 * n)
        c = np.fft.irfft(f * np.conj(f))[:9] / n
        acc += c
    acc /= reps
    k = np.arange(9)
    tgt = 0.5 * (np.abs(k - 1) ** (2 * H) - 2.0 * k ** (2 * H)
                 + (k + 1) ** (2 * H))
    return np.abs(acc - tgt).max()


def make_analyzer(H_true, mode):
    """The (m, a, H) callable run_one wants.

    m and a exist only to map x -> Z; we set them so that the offset is the
    training mean and the scale the training std, which is what run_one's
    (x - m*dt) / sqrt(m*a) reduces to. Only H differs between modes, and H is
    the whole point: 'oracle' hands over the H the data was drawn at, 'whittle'
    re-estimates it exactly as score_nmse_traffic.py does.
    """
    def analyze(x, dt):
        x = np.asarray(x, dtype=float)
        m = x.mean() / dt
        a = x.var() / m
        H = H_true if mode == "oracle" else float(
            np.clip(local_whittle(x), 0.5001, 0.9999))
        return m, a, H
    return analyze


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--h", choices=["oracle", "whittle"], default="oracle",
                    help="where the H handed to norros_f/linearP_f comes from")
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--p", type=int, default=1, help="AR order for ARFIMA(p,d,0)")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--out", default=os.path.join("results", "nmse_fgn.csv"))
    ap.add_argument("--summary-out",
                    default=os.path.join("results", "nmse_fgn_summary.txt"))
    args = ap.parse_args()

    print(f"exact fGn (Davies-Harte)  n={args.n_samples}  T={args.T} taps  "
          f"h={args.horizon}  ARFIMA p={args.p}  H={args.h}  "
          f"seeds={args.n_seeds}")

    if not args.skip_verify:
        print("\nsynthesiser check: max |mean sample acov - fGn acov| over "
              "lags 0..8, 400 reps of n=4096")
        for _, H in H_GRID:
            rng = np.random.default_rng(12345)
            print(f"  H={H:.2f}   {verify_covariance(H, rng):.2e}")

    hdr = (f"{'H_true':>7} {'seed':>5} {'H_used':>7} {'d_hat':>7} | "
           + " ".join(f"{lab:>9}" for _, lab in METHODS)
           + f" | {'acf_dev':>8}")
    print("\n" + hdr)
    print("-" * len(hdr))

    rows = []
    for alpha, H_true in H_GRID:
        analyze = make_analyzer(H_true, args.h)
        for seed in range(args.n_seeds):
            rng = np.random.default_rng(1000 * seed + int(H_true * 100))
            x = fgn_davies_harte(args.n_samples, H_true, rng)
            # Shifted off zero so run_one's x -> Z map is the same affine
            # rescaling it performs on the traffic traces. NMSE is invariant to
            # it; this only keeps the two code paths identical.
            x = 1.0 + 0.1 * x

            r = run_one(x, DT, args.T, analyze, args.p, args.horizon)
            rows.append(dict(alpha=alpha, H_true=H_true, seed=seed,
                             H_used=r["H_use"], d=r["d"], n_fc=r["n_fc"],
                             acf_dev=r["acf_dev"],
                             **{f"nmse_{lab}": r[k] for k, lab in METHODS}))
            print(f"{H_true:>7.2f} {seed:>5d} {r['H_use']:>7.3f} "
                  f"{r['d']:>7.3f} | "
                  + " ".join(f"{r[k]:>9.4f}" for k, _ in METHODS)
                  + f" | {r['acf_dev']:>8.4f}")

    df = pd.DataFrame(rows)
    lines = summary_lines(df, args)
    print("\n" + "\n".join(lines))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        df.insert(0, "run", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        df.insert(1, "T", args.T)
        df.insert(2, "horizon", args.horizon)
        df.insert(3, "p", args.p)
        df.insert(4, "h_source", args.h)
        exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        df.to_csv(args.out, mode="a", header=not exists, index=False)
        print(f"\nper-realisation rows appended -> {args.out}")
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "a") as fh:
            fh.write("\n".join(lines) + "\n\n")
        print(f"summary appended -> {args.summary_out}")


def summary_lines(df, args):
    """Per-H table plus the paired linearP_f - farima_f test.

    The paired test is the one that matters: the same realisation drives every
    predictor, so the per-seed difference removes between-seed difficulty. With
    H=oracle the theory's prediction is unambiguous -- linearP_f wins every
    seed, and any positive mean is a bug rather than a finding.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L = [f"=== {stamp}  EXACT fGn  T={args.T} h={args.horizon} p={args.p} "
         f"H={args.h}  n={args.n_samples} ===",
         "mean over seeds +- half-width of the 95% t-CI over seeds"]
    for H_true, g in df.groupby("H_true"):
        n = len(g)
        L.append(f"\nH_true={H_true:.2f}  H_used={g.H_used.mean():.3f}"
                 f"+-{ci(g.H_used)[1]:.3f}  d_hat={g.d.mean():.3f}  "
                 f"n_seeds={n}  n_fc={int(g.n_fc.iloc[0])}  "
                 f"acf_dev={g.acf_dev.mean():.4f}")
        L.append(f"{'method':>12} {'NMSE':>20}")
        for _, lab in METHODS:
            m, h = ci(g[f"nmse_{lab}"])
            L.append(f"{lab:>12} {m:>9.5f} +- {h:.5f}")
        d = 100.0 * (g["nmse_linearP_f"] - g["nmse_farima_f"]) / g["nmse_farima_f"]
        m_r, h_r = ci(d)
        t_stat, p = stats.ttest_rel(g["nmse_linearP_f"], g["nmse_farima_f"])
        L.append(f"  paired linearP_f - farima_f: {m_r:+.3f}% +- {h_r:.3f}%  "
                 f"(t={t_stat:.2f}, p={p:.4f}, linearP_f better in "
                 f"{int((d < 0).sum())}/{n})")
    return L


if __name__ == "__main__":
    main()
