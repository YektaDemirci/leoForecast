import argparse
import numpy as np

from norros_f import norros_weights, design
from farima_f import arfima_weights
from linearP_f import linearp_weights
from analyze_traffic import analyze_traffic

# torch and gen_onoff_pareto are imported inside main() only. The WiFi path
# (score_wifi.py) imports run_one from here and has no synthetic generator to
# run, so a module-level torch import would make it a hard dependency of a
# script that never touches the GPU.


ARRIVAL_DT = 0.001          # raw generator granularity [s]
SAMPLE_DT = 1            # forecasting sample granularity [s]


def fgn_autocov(k, H):
    return 0.5 * (abs(k-1)**(2*H) - 2*abs(k)**(2*H) + abs(k+1)**(2*H))

def acov(x, lag):
    if lag == 0:
        return np.var(x)
    mu = np.mean(x)
    return np.mean((x[:-lag] - mu) * (x[lag:] - mu))


def run_one(samples, dt, T, analyze=analyze_traffic, arfima_p=2, horizon=1):
    """Train/test split + both predictors. Returns a dict of diagnostics."""
    n = len(samples)
    train = samples[:int(n * 0.7)]
    # samples[0.7n:0.8n] is the validation block. Nothing here uses it -- it is
    # held out so the analytic and Informer runs score the same test block.
    test = samples[int(n * 0.8):]

    m, a, H = analyze(train, dt)
    Z_tr = (train - m * dt) / np.sqrt(m * a)
    Z_te = (test - m * dt) / np.sqrt(m * a)

    # H for the kernel is the estimator's answer to "which H describes this
    # series", full stop. There used to be a --fit-hurst mode that instead
    # grid-searched H to minimize the kernel's OWN forecast error; it is gone.
    # It only ever tuned Norros and linearP -- farima fits its d by MLE and
    # could not be tuned the same way -- so it tilted the very comparison this
    # script exists to make, and is indistinguishable from cherry-picking.
    H_use = H

    # --- Norros g_T kernel, integrated over each unit cell (norros_f.py) ---
    w_g = norros_weights(T, H_use, horizon=float(horizon))

    # --- linearP (model-based): same H, but the exactly optimal DISCRETE
    # projection rather than a discretized continuous kernel (linearP_f.py).
    # It is an upper bound on what g_T can reach on this grid, so gt - lp is
    # the kernel's discretization penalty.
    w_lp = linearp_weights(T, H_use, horizon=horizon)

    # --- ARFIMA(p,d,0) fitted once on the training set (farima_f.py) ---
    w_far, d_hat, _phi = arfima_weights(Z_tr, T, p=arfima_p, horizon=horizon)

    # --- baselines, retained but not part of the comparison ---
    # row = np.array([acov(Z_tr, k) for k in range(T)])
    # w_lp = np.linalg.solve(scipy.linalg.toeplitz(row) + 1e-8 * np.eye(T),
    #                        np.array([acov(Z_tr, 1 + k) for k in range(T)]))
    # row_te = np.array([acov(Z_te, k) for k in range(T)])
    # w_or = np.linalg.solve(scipy.linalg.toeplitz(row_te) + 1e-8 * np.eye(T),
    #                        np.array([acov(Z_te, 1 + k) for k in range(T)]))

    # Same windowing helper select_H scores with, so selection and evaluation
    # cannot drift apart.
    win, truth = design(Z_te, T, horizon)

    def nmse(w):
        return np.mean((truth - win @ w) ** 2) / np.var(truth)

    # How far the data is from fGn: max |empirical ACF - fGn ACF| over lags 1-8,
    # at the *estimated* H. Large values mean the single-parameter fGn model is
    # misspecified, which is exactly what ARFIMA's p AR terms can absorb.
    s = np.array([acov(Z_tr, k) for k in range(9)])
    s = s / s[0]
    th = np.array([fgn_autocov(k, H) / fgn_autocov(0, H) for k in range(9)])

    w_naive = horizon * np.eye(T)[0]

    return dict(H=H, H_use=H_use,
                d=d_hat, n_train=len(Z_tr), n_test=len(Z_te),
                n_fc=len(truth), gt=nmse(w_g), far=nmse(w_far),
                lp=nmse(w_lp),
                # persistence: the most recent cell, repeated `horizon` times
                naive=nmse(w_naive),
                acf_dev=np.abs(s[1:] - th[1:]).max(),
                # --- additions, purely so a caller can re-derive these same
                # predictions in units other than Z. Nothing above depends on
                # them and no existing key changed value.
                #
                # Z = (x - z_offset) / z_scale, so a predicted h-step SUM in Z
                # maps back as  x_sum = z_scale * pred + horizon * z_offset;
                # design() puts the truth for row i at test[test_start + i + T
                # : ... + horizon], which is what score_wifi.py uses to add the
                # seasonal/trend component back and score in the raw units.
                weights=dict(gt=w_g, far=w_far, lp=w_lp, naive=w_naive),
                z_offset=m * dt, z_scale=float(np.sqrt(m * a)),
                T=T, horizon=horizon, test_start=int(n * 0.8), win=win)


def main():
    import torch
    from gen_onoff_pareto import simulate_onoff_aggregate

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sources", type=int, default=500)
    ap.add_argument("--alpha", type=float, default=1.04,
                    help="Pareto tail index for both ON and OFF; H=(3-alpha)/2")
    ap.add_argument("--rate-on", type=float, default=1e6, help="bits/s per ON source")
    ap.add_argument("--minutes", type=float, default=15.0)
    ap.add_argument("--T", type=int, default=48, help="predictor taps (past samples)")
    ap.add_argument("--horizon", type=int, default=1,
                    help="forecast lead h: predict the cumulative demand over "
                         "the next h samples (h=1 is the one-step case)")
    ap.add_argument("--p", type=int, default=2,
                    help="AR order for the ARFIMA(p,d,0) in farima_f.py")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[101, 202, 303, 404, 505, 606, 707, 808, 909, 1010])
    ap.add_argument("--n-seeds", type=int, default=None,
                    help="use this many generated seeds instead of --seeds; "
                         "n=28 gives 80%% power at the effect size seen so far, "
                         "n=50 gives ~97%%")
    args = ap.parse_args()
    if args.n_seeds is not None:
        args.seeds = [101 * (i + 1) for i in range(args.n_seeds)]
    if args.horizon < 1:
        ap.error("--horizon must be >= 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    horizon = args.minutes * 60.0
    G = int(round(SAMPLE_DT / ARRIVAL_DT))
    n_samples = int(round(horizon / SAMPLE_DT))
    H_theory = (3.0 - args.alpha) / 2.0

    analyze = analyze_traffic
    print(f"device={device}  X={args.n_sources}  alpha={args.alpha} "
          f"(H_theory={H_theory:.3f})  xm_on=xm_off=0.001  "
          f"H estimator: variance-time")
    print(f"{args.minutes:g} min @ {ARRIVAL_DT*1e3:.0f}ms -> aggregate G={G} "
          f"-> {n_samples} samples of {SAMPLE_DT:g}s   T={args.T} taps\n"
          f"target: cumulative demand over the next h={args.horizon} samples "
          f"({args.horizon * SAMPLE_DT:g}s ahead)\n")

    hdr = f"{'seed':>6} {'H_est':>7} {'d_hat':>7} | " \
          f"{'norros_f':>9} {'linearP_f':>9} {'farima_f':>9} {'naive':>8} | " \
          f"{'gap%':>7} {'acf_dev':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for seed in args.seeds:
        rate, _ = simulate_onoff_aggregate(
            args.n_sources, horizon, ARRIVAL_DT, args.alpha, args.alpha,
            xm_on=0.001, xm_off=0.001, rate_on=args.rate_on,
            device=device, seed=seed, stationary_start=True)
        demand = rate.cpu().numpy() * ARRIVAL_DT
        del rate
        n = len(demand) // G * G
        samples = demand[:n].reshape(-1, G).sum(axis=1)

        r = run_one(samples, SAMPLE_DT, args.T, analyze, args.p, args.horizon)
        r["seed"] = seed
        r["gap"] = 100.0 * (r["gt"] - r["far"]) / r["far"]
        rows.append(r)
        print(f"{seed:>6} {r['H']:>7.3f} {r['d']:>7.3f} | "
              f"{r['gt']:>9.4f} {r['lp']:>9.4f} {r['far']:>9.4f} "
              f"{r['naive']:>8.4f} | {r['gap']:>6.1f}% {r['acf_dev']:>8.4f}")

    print("-" * len(hdr))
    agg = lambda k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows], ddof=1))
    print(f"{'mean':>6} {agg('H')[0]:>7.3f} "
          f"{agg('d')[0]:>7.3f} | {agg('gt')[0]:>9.4f} {agg('lp')[0]:>9.4f} "
          f"{agg('far')[0]:>9.4f} {agg('naive')[0]:>8.4f} | "
          f"{agg('gap')[0]:>6.1f}% {agg('acf_dev')[0]:>8.4f}")
    print(f"{'sd':>6} {agg('H')[1]:>7.3f} "
          f"{agg('d')[1]:>7.3f} | {agg('gt')[1]:>9.4f} {agg('lp')[1]:>9.4f} "
          f"{agg('far')[1]:>9.4f} {agg('naive')[1]:>8.4f} | "
          f"{agg('gap')[1]:>6.1f}% {agg('acf_dev')[1]:>8.4f}")

    r0 = rows[0]
    print(f"\nper seed: {r0['n_train']} train / {r0['n_test']} test samples "
          f"-> {r0['n_fc']} forecasts at h={args.horizon}")

    summarize(rows)


def ci(x, conf=0.95):
    """Mean and t-based CI half-width. Each seed is one independent replicate."""
    from scipy import stats
    x = np.asarray(x, dtype=float)
    n = len(x)
    half = stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return x.mean(), half


def summarize(rows, conf=0.95):
    """Marginal CIs per method, then the PAIRED method comparison.

    Seeds are the unit of analysis (n = number of seeds), not individual
    forecasts: errors within one realization are strongly correlated, so
    pooling forecasts would understate the true uncertainty by a large factor.
    """
    from scipy import stats
    gt = np.array([r["gt"] for r in rows])       # norros_f
    lp = np.array([r["lp"] for r in rows])       # linearP_f
    far = np.array([r["far"] for r in rows])     # farima_f
    n = len(rows)

    print(f"\n--- across {n} seeds, {int(conf*100)}% CI (t, {n-1} df) ---")
    for name, v in [("norros_f g_T", gt), ("linearP", lp), ("farima_f", far),
                    ("naive", np.array([r["naive"] for r in rows]))]:
        m, h = ci(v, conf)
        print(f"  {name:>13}  NMSE = {m:.4f} +/- {h:.4f}   [{m-h:.4f}, {m+h:.4f}]")
    m, h = ci([r["H"] for r in rows], conf)
    print(f"  {'H_est':>13}       = {m:.4f} +/- {h:.4f}   [{m-h:.4f}, {m+h:.4f}]")

    d_rel = 100.0 * (gt - far) / far

    # g_T vs linearP isolates one thing only: both use the same H and the
    # same fGn model, so the difference is purely the cost of discretizing a
    # continuous-time kernel onto unit cells. Theory says 0.1-0.3% at these H.
    m_d, h_d = ci(100.0 * (gt - lp) / lp, conf)
    print(f"\n--- g_T discretization penalty (norros_f - linearP) ---")
    print(f"  relative  {m_d:+.2f}% +/- {h_d:.2f}%   "
          f"[{m_d-h_d:+.2f}%, {m_d+h_d:+.2f}%]   (theory: +0.1 to +0.3%)")

    # The per-seed gap distribution is skewed (a few hard realizations dominate
    # the mean), so back the t-test with distribution-free statistics.
    w_stat, w_p = stats.wilcoxon(gt, far)
    print(f"  median relative {np.median(d_rel):+.2f}%   "
          f"Wilcoxon signed-rank: W = {w_stat:.0f}, p = {w_p:.4f}")

    # Bootstrap cross-check: does not assume the seed means are normal.
    rng = np.random.default_rng(0)
    bs = np.array([d_rel[rng.integers(0, n, n)].mean() for _ in range(20000)])
    lo, hi = np.percentile(bs, [100*(1-conf)/2, 100*(1+conf)/2])
    print(f"  bootstrap CI on relative: [{lo:+.2f}%, {hi:+.2f}%]  (20k resamples)")
    print("\nnegative => norros_f wins. CI excluding 0 => difference is resolved.")


if __name__ == "__main__":
    main()
