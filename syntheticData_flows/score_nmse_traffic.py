import argparse
import datetime
import os

import numpy as np
import pandas as pd
from scipy import stats

from single_cell_1s import run_one
from scoring import raw_space_nmse, RAW_METHODS
from h_estimators import local_whittle, analyze_fit

DATA_DIR = "./nmse_traffic"
DT = 1.0


def report_h(man, series, args, conf=0.95):

    print("\nH on the training block (first 70%). H_fit is what norros_f and "
          "linearP_f are built from -- the minimiser of linearP's own forecast "
          "error, a fitted filter parameter, not a Hurst estimate; farima_f "
          "fits its own d.")
    print("H_whittle is the local Whittle estimate on the same training block, "
          "reported as the Hurst exponent of the trace. Nothing is built from "
          "it, so read its gap against H_theory as estimator error and H_fit's "
          "as fGn misfit, not as the same quantity.")
    hdr = (f"{'file':>24} {'H_theory':>8} {'H_fit':>8} {'H_gap':>7}"
           f" {'H_whittle':>9} {'H_w_gap':>8}")
    print(hdr)
    print("-" * len(hdr))
    analyze = analyze_fit(args.T, args.horizon)
    hs, hs_w = {}, {}
    for _, meta in man.iterrows():
        x = series[meta.file]
        tr = x[:int(len(x) * 0.7)]
        _, _, H = analyze(tr, args.dt)
        hs[meta.file] = H
        # Unclipped: nothing is built from it, and clipping to the kernel's
        # (0.5, 1) range would hide how close to 1 the estimate really lands.
        H_w = float(local_whittle(tr))
        hs_w[meta.file] = H_w
        print(f"{meta.file:>24} {meta.H_theory:>8.2f} {H:>8.3f} "
              f"{H - meta.H_theory:>+7.3f} {H_w:>9.3f} "
              f"{H_w - meta.H_theory:>+8.3f}")
    print("-" * len(hdr))
    for alpha, g in man.groupby("alpha"):
        v = np.array([hs[f] for f in g.file])
        m, h = ci(v, conf)
        Ht = g.H_theory.iloc[0]
        print(f"  alpha={alpha:.2f}  H_theory={Ht:.2f}  "
              f"H_fit     = {m:.3f} +- {h:.3f}  [{m - h:.3f}, {m + h:.3f}]  "
              f"gap {m - Ht:+.3f}")
        v = np.array([hs_w[f] for f in g.file])
        m, h = ci(v, conf)
        print(f"  {'':>10}  {'':>13}  "
              f"H_whittle = {m:.3f} +- {h:.3f}  [{m - h:.3f}, {m + h:.3f}]  "
              f"bias {m - Ht:+.3f}")
    return hs, hs_w


def ci(x, conf=0.95):
    x = np.asarray(x, dtype=float)
    n = len(x)
    return x.mean(), stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR)
    ap.add_argument("--dt", type=float, default=DT,
                    help="bin width [s] of the traces; must match the "
                         "generator's --dt (MASE is invariant to it, but the "
                         "(m, a) diagnostics are not)")
    ap.add_argument("--col", default="OT")
    ap.add_argument("--T", type=int, default=48, help="predictor taps")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--p", type=int, default=2, help="AR order for ARFIMA(p,d,0)")
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--per-fc-out", default=None,
                    help="directory for the per-forecast dumps (one CSV per "
                         "trace); skipped if not given")
    ap.add_argument("--out", default=os.path.join("results", "mase_traffic.csv"),
                    help="append one row per trace to this CSV")
    ap.add_argument("--res-txt",
                    default=os.path.join("results", "mase_traffic_res.txt"),
                    help="append one whitespace-aligned line per forecaster "
                         "per trace (FORECASTER LOOKBACK H_EST DATANAME ALPHA "
                         "SEED MSE MASE) to this file. Long format, same "
                         "convention as score_wifi.py's --res-txt. Pass '' to "
                         "skip.")
    ap.add_argument("--summary-out",
                    default=os.path.join("results", "mase_traffic_summary.txt"),
                    help="write the per-alpha mean +- CI table here")
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))
    if args.alphas:
        man = man[man.alpha.isin(args.alphas)]
    if args.seeds is not None:
        man = man[man.seed.isin(args.seeds)]

    print(f"{len(man)} traces from {args.data_dir}  T={args.T} taps  "
          f"h={args.horizon}  ARFIMA p={args.p}  H=fit (kernels), "
          f"whittle (reported)")

    series = {f: pd.read_csv(os.path.join(args.data_dir, f))[args.col]
              .to_numpy(dtype=float) for f in man.file}
    _, hs_w = report_h(man, series, args)
    hdr = (f"{'file':>24} {'H_th':>5} {'H_fit':>7} {'H_wht':>7} {'d_hat':>7} | "
           + " ".join(f"{lab:>9}" for _, lab in RAW_METHODS)
           + f" | {'acf_dev':>8}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for _, meta in man.iterrows():
        x = series[meta.file]

        r = run_one(x, args.dt, args.T, analyze_fit(args.T, args.horizon),
                    args.p, args.horizon)
        # comp = 0: nothing was removed, so the traffic-units inversion is just
        # the Z -> original-units rescaling. Reusing scoring.raw_space_nmse
        # keeps the MSE/MASE definition byte-identical to the WiFi table --
        # including the MASE naive scale, which is taken on the same test-window
        # truth there and here.
        metrics, idx, truth, preds = raw_space_nmse(r, np.zeros_like(x), x)

        rows.append(dict(file=meta.file, alpha=meta.alpha, seed=int(meta.seed),
                         H_theory=meta.H_theory, H_fit=r["H"],
                         H_whittle=hs_w[meta.file], d=r["d"],
                         n_train=r["n_train"], n_test=r["n_test"],
                         n_fc=r["n_fc"], acf_dev=r["acf_dev"],
                         **{f"mse_{lab}": metrics[k]["mse"]
                            for k, lab in RAW_METHODS},
                         **{f"mase_{lab}": metrics[k]["mase"]
                            for k, lab in RAW_METHODS}))
        print(f"{meta.file:>24} {meta.H_theory:>5.2f} {r['H']:>7.3f} "
              f"{hs_w[meta.file]:>7.3f} {r['d']:>7.3f} | "
              + " ".join(f"{metrics[k]['mase']:>9.4f}" for k, _ in RAW_METHODS)
              + f" | {r['acf_dev']:>8.4f}")

        if args.per_fc_out:
            os.makedirs(args.per_fc_out, exist_ok=True)
            fc = pd.DataFrame(dict(idx=idx, truth=truth))
            for k, lab in RAW_METHODS:
                fc[lab] = preds[k]
            fc.to_csv(os.path.join(args.per_fc_out, f"fc_{meta.file}"),
                      index=False)

    df = pd.DataFrame(rows)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.insert(0, "run", stamp)
    df.insert(1, "T", args.T)
    df.insert(2, "horizon", args.horizon)
    df.insert(3, "p", args.p)
    # Kept as a constant column: the running CSV predates the single-arm flow
    # and rows from the old vt/whittle/theory arms are still in it.
    df.insert(4, "h_estimator", "fit")

    lines = summary_lines(df, args)
    print("\n" + "\n".join(lines))

    if args.res_txt:
        append_res_txt(args.res_txt, df, args)
        print(f"per-forecaster MASE lines appended -> {args.res_txt}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        df.to_csv(args.out, mode="a", header=not exists, index=False)
        print(f"\nper-trace rows appended -> {args.out}")
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "a") as fh:
            fh.write("\n".join(lines) + "\n\n")
        print(f"summary appended -> {args.summary_out}")


RES_TXT_HDR = (f"{'FORECASTER':<12} {'LOOKBACK':>8} {'H_EST':>7} "
               f"{'DATANAME':>24} {'ALPHA':>6} {'SEED':>5} "
               f"{'MSE':>13} {'MASE':>9}")


def append_res_txt(path, df, args):
    """One line per forecaster per trace, long format.

    Deliberately not a CSV and separate from the per-alpha summary: this is the
    running scoreboard across lookbacks, horizons and H estimators, so adding a
    predictor adds rows rather than columns -- same convention as
    score_wifi.append_res_txt, with the alpha/seed keys these traces need.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a") as fh:
        if new:
            fh.write(RES_TXT_HDR + "\n")
        for _, r in df.iterrows():
            for _, lab in RAW_METHODS:
                fh.write(f"{lab.upper():<12} {args.T:>8d} "
                         f"{'fit':>7} {r.file:>24} "
                         f"{r.alpha:>6.2f} {int(r.seed):>5d} "
                         f"{r[f'mse_{lab}']:>13.4e} {r[f'mase_{lab}']:>9.4f}\n")


def summary_lines(df, args):
    """Per-alpha mean +- 95% t-CI half-width over seeds, MSE and MASE"""
    L = [f"=== {df['run'].iloc[0]}  T={args.T} h={args.horizon} p={args.p} "
         f"H=fit  {args.data_dir} ===",
         "mean over seeds +- half-width of the 95% t-CI over seeds "
         "(seeds are the unit of analysis)"]
    for alpha, g in df.groupby("alpha"):
        n = len(g)
        L.append(f"\nalpha={alpha:.2f}  H_theory={g.H_theory.iloc[0]:.2f}  "
                 f"H_fit={g.H_fit.mean():.3f}+-{ci(g.H_fit)[1]:.3f}  "
                 f"H_whittle={g.H_whittle.mean():.3f}"
                 f"+-{ci(g.H_whittle)[1]:.3f}  "
                 f"n_seeds={n}  n_fc={int(g.n_fc.iloc[0])}")
        L.append(f"{'method':>12} {'MSE':>24} {'MASE':>20}")
        for _, lab in RAW_METHODS:
            mm, mh = ci(g[f"mse_{lab}"])
            nm, nh = ci(g[f"mase_{lab}"])
            L.append(f"{lab:>12} {mm:>13.4e} +- {mh:<8.2e} "
                     f"{nm:>9.4f} +- {nh:.4f}")
    return L


if __name__ == "__main__":
    main()
