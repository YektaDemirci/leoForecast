import argparse
import datetime
import os

import numpy as np
import pandas as pd

from single_cell_1s import run_one
from scoring import RAW_METHODS, raw_space_nmse
from h_estimators import analyze_fit
from deseason import deseasonalise_parts


HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "wifiData.csv")
FC_ANALYTIC = "results/fc_analytic.csv"     # per-forecast dump, traffic units
SAMPLE_DT = 600.0           # 10-minute grid [s]
SLOTS_PER_DAY = 144



def load_wifi(path=CSV):
    """Read, and repair the DST fallback at 2001-10-28.

    The raw file steps backwards 50 minutes there, leaving 6 duplicate
    timestamps and a non-monotonic index; sorting and de-duplicating restores
    the uniform 600 s grid every lag-based statistic assumes. Same repair as
    analyze_wifi_hurst.load.
    """
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").drop_duplicates("date").reset_index(drop=True)


def where(df, ts, n):
    """Which of the 70/10/20 blocks the timestamp falls in."""
    i = int(np.searchsorted(df.date.values, np.datetime64(ts)))
    if i < int(n * 0.7):
        return i, "train"
    if i < int(n * 0.8):
        return i, "val"
    return i, "test"


def score(name, samples, args, analyze):
    r = run_one(samples, SAMPLE_DT, args.T, analyze, args.p, args.horizon)
    return r


RES_TXT_HDR = f"{'FORECASTER':<12} {'LOOKBACK':>8} {'DATANAME':>20} {'SERIES':>15} {'MASE':>9}"


def append_res_txt(path, metrics, T, csv_path, series):
    """Append one line per forecaster to a flat, greppable results file.

    Deliberately not a CSV: this is the running scoreboard across datasets,
    lookbacks and series, so it stays in long format (one row per forecaster)
    rather than growing a column every time a predictor is added.
    """
    name = os.path.basename(csv_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a") as fh:
        if new:
            fh.write(RES_TXT_HDR + "\n")
        for key, label in RAW_METHODS:
            fh.write(f"{label.upper():<12} {T:>8d} {name:>20} {series:>15} "
                     f"{metrics[key]['mase']:>9.4f}\n")
    print(f"MASE lines appended -> {path}")


def report_raw(metrics, r_des, idx):
    """The (B) table: MSE in traffic units, the headline comparison.

    MSE is the primary column now. MASE is kept beside it only because it is
    dimensionless -- MSE on a series of ~1e12 is ~1e23, which is awkward to
    read and impossible to compare against a differently-scaled trace. MASE
    normalises the MAE by the mean absolute one-step change of the truth, so
    1.0 is exactly persistence.
    """
    hdr = f"{'method':>14} {'MSE':>13} {'MASE':>9}"
    print(f"\nTRAFFIC units -- forecasts re-seasonalised, scored against raw "
          f"OT over buckets [{idx[0]}, {idx[-1]}]")
    print(hdr)
    print("-" * len(hdr))
    for key, label in RAW_METHODS:
        m = metrics[key]
        print(f"{label:>14} {m['mse']:>13.4e} {m['mase']:>9.4f}")
    print("-" * len(hdr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=CSV)
    ap.add_argument("--col", default="OT")
    ap.add_argument("--deseason-csv", default=None,
                     help="if given, read the deseasonalised series straight "
                          "from this file's --deseason-col instead of fitting "
                          "the causal time-of-day/time-of-week profiles here "
                          "(e.g. dataset/deseasonalisedWifiDataset.csv)")
    ap.add_argument("--deseason-col", default="OT_deseason")
    ap.add_argument("--no-deseason", action="store_true",
                     help="feed the RAW series to the "
                          "predictors -- no seasonal profiles, no detrend. The "
                          "removed component is then identically zero, so the "
                          "traffic-units table below needs no re-seasonalising "
                          "and is directly comparable to a deseasonalised run "
                          "over the same buckets.")
    ap.add_argument("--T", type=int, default=48, help="predictor taps")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--p", type=int, default=1, help="AR order for ARFIMA(p,d,0)")
    ap.add_argument("--fc-out", default=FC_ANALYTIC,
                     help="per-forecast dump (traffic units, keyed by global "
                          "bucket index). Give raw and deseasonalised runs "
                          "different paths so compare_forecasts.py can put all "
                          "of them in one table.")
    ap.add_argument("--res-txt", default=os.path.join("results", "res.txt"),
                     help="append one whitespace-aligned line per forecaster "
                          "(FORECASTER LOOKBACK DATANAME SERIES MASE) to this "
                          "file. Long format, unlike the wide one-row-per-run "
                          "--out CSV. Pass '' to skip.")
    ap.add_argument("--out", default=None,
                     help="append the raw/deseasonalised MASE rows to this CSV "
                          "(one row per series); skipped if not given")
    args = ap.parse_args()

    # The H handed to norros_f/linearP_f is always H_fit: the H minimising
    # linearP's own training forecast error -- a fitted filter parameter, NOT a
    # Hurst estimate. farima_f never uses it, it fits its own d by MLE. Matches
    # syntheticData_flows/nmseTrafficRuns.sh, which runs the same arm.
    analyze = analyze_fit(args.T, args.horizon)
    df = load_wifi(args.csv)
    n = len(df)
    n_train, n_val = int(n * 0.7), int(n * 0.8)

    print(f"{args.csv}: {n} samples @ {SAMPLE_DT:g}s, "
          f"{df.date.iloc[0]} .. {df.date.iloc[-1]}")
    print(f"split: train [0,{n_train}) -> {df.date.iloc[n_train - 1]}   "
          f"val [{n_train},{n_val}) -> {df.date.iloc[n_val - 1]}   "
          f"test [{n_val},{n})")


    raw = df[args.col].to_numpy(dtype=float)
    if args.no_deseason:
        # Nothing is removed, so comp is zero and des IS the traffic series.
        # That makes the traffic-units table an identity rather than an
        # inversion, and the residual-units table above collapses onto it.
        xf = raw
        des, comp = xf, np.zeros_like(xf)
    elif args.deseason_csv:
        dd = pd.read_csv(args.deseason_csv, parse_dates=["date"])
        dd = dd.sort_values("date").drop_duplicates("date")
        merged = df[["date"]].merge(dd[["date", args.deseason_col]], on="date",
                                     how="left")
        if merged[args.deseason_col].isna().any():
            raise SystemExit(f"{args.deseason_csv} is missing dates present in "
                              f"{args.csv}")
        des = merged[args.deseason_col].to_numpy(dtype=float)
        # The file carries only the residual, so the removed component is
        # recovered by subtraction against the raw column -- the same series
        # the built-in transform would have differenced against.
        xf = raw
        comp = xf - des
    else:
        des, xf, comp = deseasonalise_parts(df, n_train, args.col)
        # des is zero-mean by construction, and run_one standardises by
        # (m*dt, sqrt(m*a)) estimated from the mean -- which blows up at m ~ 0.
        # The train mean shifts it back onto the traffic scale; taking the same
        # constant back out of comp keeps x == des + comp exact, so the
        # traffic-units inversion is untouched. Z comes out as (x - mean)/scale
        # either way, so no forecast moves because of the shift, and
        # analyze_fit standardises internally, so H_fit does not move either.
        shift = float(np.mean(xf[:n_train]))
        des, comp = des + shift, comp - shift
    label = "raw" if args.no_deseason else "deseasonalised"
    r_des = score(label, des, args, analyze)

    metrics, idx, truth, preds = raw_space_nmse(r_des, comp, xf)


    report_raw(metrics, r_des, idx)

    # Per-forecast dump, keyed by global bucket index so compare_forecasts.py
    # can put this and the Informer run on the same buckets. Overwritten each
    # run: it describes this run only, unlike --out which accumulates.
    #
    # `season` is the component this run added back -- identically zero under
    # --no-deseason. compare_forecasts.py checks it against deseason.py, which
    # is the only check that catches a re-seasonalisation done with a stale
    # deseasonalised csv: `truth` is the raw series in both variants, so it
    # agrees even then, and a date-alignment assert passes because a stale file
    # keeps its dates and row count.
    # Summed over the SAME h buckets raw_space_nmse sums the truth over, so at
    # horizon > 1 this stays the quantity actually added to the forecast rather
    # than the first bucket of it.
    _cc = np.concatenate(([0.0], np.cumsum(comp)))
    season = _cc[idx + args.horizon] - _cc[idx]
    fc = pd.DataFrame(dict(idx=idx, date=df.date.to_numpy()[idx], truth=truth,
                           season=season))
    for key, lab in RAW_METHODS:
        fc[lab] = preds[key]
    os.makedirs(os.path.dirname(args.fc_out) or ".", exist_ok=True)
    fc.to_csv(args.fc_out, index=False)
    print(f"per-forecast values -> {args.fc_out}")

    if args.res_txt:
        # Long format: one line per forecaster, traffic-units MASE (the (B)
        # table above). SERIES is carried because raw and deseasonalised runs
        # share the same dataset name and lookback and would otherwise be
        # indistinguishable once appended.
        append_res_txt(args.res_txt, metrics, args.T, args.csv, label)

    if args.out:
        # Appended, not overwritten, so repeated runs (different --T, --horizon,
        # ...) accumulate in one file -- same convention as score_analytic.py.
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        r = r_des
        out = pd.DataFrame([
            dict(run=stamp, csv=args.csv, series=label, T=args.T,
                 horizon=args.horizon, p=args.p, H=r["H"], d=r["d"], n_train=r["n_train"], n_test=r["n_test"],
                 n_fc=r["n_fc"], norros_f=r["gt"], linearP_f=r["lp"],
                 farima_f=r["far"], naive=r["naive"], acf_dev=r["acf_dev"],
                 # traffic-units (B) columns; the four above stay in residual
                 # units so previously recorded runs remain interpretable.
                 # MSE first -- that is the headline now -- MASE retained
                 # because MSE alone is unreadable at ~1e23. Derived from
                 # RAW_METHODS so the schema cannot drift from the table.
                 **{f"mse_{lab}": metrics[k]["mse"] for k, lab in RAW_METHODS},
                 **{f"mase_{lab}": metrics[k]["mase"] for k, lab in RAW_METHODS})])
        exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        if exists:
            old = pd.read_csv(args.out, nrows=0).columns.tolist()
            if old != out.columns.tolist():
                raise SystemExit(
                    f"{args.out} has columns {old}, but this run writes "
                    f"{out.columns.tolist()}. Move or delete it, or pass a "
                    f"different --out.")
        out.to_csv(args.out, mode="a", header=not exists, index=False)
        print(f"\nMASE appended -> {args.out}")


if __name__ == "__main__":
    main()
