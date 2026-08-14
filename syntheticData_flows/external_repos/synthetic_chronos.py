import argparse
import datetime
import os

import numpy as np
import pandas as pd
from scipy import stats

DATA_DIR = "./nmse_traffic"
DT = 1.0
FREQ = "1s"          # must match DT
MODEL_PATH = "amazon/chronos-bolt-tiny"
TRAIN_FRAC = 0.70
TEST_FRAC = 0.20


def ci(x, conf=0.95):
    """Mean and t-based CI half-width. Each seed is one independent replicate."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return x.mean(), stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def forecast_trace(x, dates, T, horizon, fine_tune, model_path, verbosity=0):
    """Zero-shot or fine-tuned Chronos-Bolt over one trace's test block.

    Returns (idx, truth, pred) with idx the GLOBAL sample index of each
    forecast target -- the same keying score_nmse_traffic.py's per-forecast
    dumps use, so the two can be joined bucket by bucket.
    """
    from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

    n = len(x)
    n_train = int(n * TRAIN_FRAC)
    test_start = int(n * (1.0 - TEST_FRAC))     # == int(0.8n), the analytic split

    # Train-only standardisation: the test block must not touch the scaler.
    mu, sd = x[:n_train].mean(), x[:n_train].std()
    z = (x - mu) / sd

    df = pd.DataFrame({"item_id": "S", "date": dates, "OT": z})
    ts = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id",
                                             timestamp_column="date")

    # Fitted on the training block only. The 10% validation block between train
    # and test is left untouched, exactly as the analytic path leaves it.
    predictor = TimeSeriesPredictor(
        target="OT", prediction_length=horizon, freq=FREQ, verbosity=verbosity,
    ).fit(
        ts.iloc[:n_train],
        hyperparameters={"Chronos": {"model_path": model_path,
                                     "fine_tune": fine_tune,
                                     "context_length": T}},
    )

    # Windows aligned with single_cell_1s.design(): target i is at
    # test_start + T + i, context is the T samples before it. The last window
    # must leave `horizon` samples inside the trace.
    n_fc = n - (test_start + T) - horizon + 1
    batch = []
    for i in range(n_fc):
        j = test_start + T + i                  # first target of window i
        batch.append(pd.DataFrame({"item_id": f"W_{i}",
                                   "date": dates[j - T:j],
                                   "OT": z[j - T:j]}))
    bts = TimeSeriesDataFrame.from_data_frame(
        pd.concat(batch, ignore_index=True),
        id_column="item_id", timestamp_column="date")

    pred = predictor.predict(bts)
    y = np.array([pred.loc[f"W_{i}"]["mean"].to_numpy()[:horizon]
                  for i in range(n_fc)])
    y = y * sd + mu                             # back to traffic units

    idx = (test_start + T + np.arange(n_fc))[:, None] + np.arange(horizon)[None, :]
    truth = x[idx]
    # h > 1 targets the CUMULATIVE demand over the next h samples, as the
    # analytic predictors do, so the two are scored on the same quantity.
    return idx[:, 0], truth.sum(axis=1), y.sum(axis=1)


def mase(pred, true):
    """MASE = MAE / mean absolute one-step change of `true`.

        MASE = [1/N sum |y - yhat|] / [1/(T-1) sum_{t=2..T} |y_t - y_{t-1}|]

    The naive scale is taken on the series being scored (the test-window truth),
    so 1.0 is exactly the one-step persistence forecast. Same definition as
    scoring.naive_scale in the analytic flow, so the scoreboards concatenate.
    """
    true = np.asarray(true, dtype=float)
    scale = np.mean(np.abs(np.diff(true)))
    return float(np.mean(np.abs(true - np.asarray(pred, dtype=float))) / scale)


RES_TXT_HDR = (f"{'FORECASTER':<14} {'LOOKBACK':>8} {'DATANAME':>24} "
               f"{'ALPHA':>6} {'SEED':>5} {'MSE':>13} {'MASE':>9}")


def append_res_txt(path, df, args):
    """One line per variant per trace, same long format as
    score_nmse_traffic.append_res_txt so both scoreboards concatenate."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a") as fh:
        if new:
            fh.write(RES_TXT_HDR + "\n")
        for _, r in df.iterrows():
            fh.write(f"{('CHRONOS_' + r.variant).upper():<14} {args.T:>8d} "
                     f"{r.file:>24} {r.alpha:>6.2f} {int(r.seed):>5d} "
                     f"{r.mse:>13.4e} {r.mase:>9.4f}\n")


def summary_lines(df, args):
    """Per-alpha mean +- 95% t-CI half-width over seeds, MSE and MASE, plus the
    paired fine-tuned vs zero-shot comparison the shared seeds make possible."""
    L = [f"=== {df['run'].iloc[0]}  CHRONOS {args.model_path}  T={args.T} "
         f"h={args.horizon}  {args.data_dir} ===",
         "mean over seeds +- half-width of the 95% t-CI over seeds "
         "(seeds are the unit of analysis)"]
    for alpha, g in df.groupby("alpha"):
        L.append(f"\nalpha={alpha:.2f}  H_theory={g.H_theory.iloc[0]:.2f}  "
                 f"n_seeds={g.seed.nunique()}  n_fc={int(g.n_fc.iloc[0])}")
        L.append(f"{'variant':>14} {'MSE':>24} {'MASE':>20}")
        for variant, gv in g.groupby("variant"):
            mm, mh = ci(gv.mse)
            nm, nh = ci(gv.mase)
            L.append(f"{variant:>14} {mm:>13.4e} +- {mh:<8.2e} "
                     f"{nm:>9.4f} +- {nh:.4f}")
        # Paired: the same realisation drives both variants, so the per-seed
        # difference removes between-seed difficulty.
        w = g.pivot_table(index="seed", columns="variant", values="mase")
        if {"zeroshot", "finetuned"} <= set(w.columns) and len(w) > 1:
            d = 100.0 * (w["finetuned"] - w["zeroshot"]) / w["zeroshot"]
            m_r, h_r = ci(d)
            t_stat, p = stats.ttest_rel(w["finetuned"], w["zeroshot"])
            L.append(f"  paired finetuned - zeroshot: {m_r:+.2f}% +- {h_r:.2f}%"
                     f"  (t={t_stat:.2f}, p={p:.4f}, fine-tuning better in "
                     f"{int((d < 0).sum())}/{len(d)})")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="directory holding the trace CSVs and manifest.csv")
    ap.add_argument("--col", default="OT")
    ap.add_argument("--T", type=int, default=48, help="context length")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--model-path", default=MODEL_PATH)
    ap.add_argument("--variants", nargs="+", default=["zeroshot", "finetuned"],
                    choices=["zeroshot", "finetuned"])
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--per-fc-out", default=None,
                    help="directory for per-forecast dumps (one CSV per trace "
                         "per variant); skipped if not given")
    ap.add_argument("--out",
                    default=os.path.join("results", "mase_traffic_chronos.csv"))
    ap.add_argument("--res-txt",
                    default=os.path.join("results", "mase_traffic_res.txt"),
                    help="appended in the same long format the analytic run "
                         "uses, so one file holds every forecaster")
    ap.add_argument("--summary-out",
                    default=os.path.join("results",
                                         "mase_traffic_chronos_summary.txt"))
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))
    if args.alphas:
        man = man[man.alpha.isin(args.alphas)]
    if args.seeds is not None:
        man = man[man.seed.isin(args.seeds)]

    print(f"{len(man)} traces from {args.data_dir}  model={args.model_path}  "
          f"T={args.T} context  h={args.horizon}  variants={args.variants}")
    hdr = f"{'file':>24} {'variant':>10} {'n_fc':>6} {'MSE':>13} {'MASE':>9}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for _, meta in man.iterrows():
        d = pd.read_csv(os.path.join(args.data_dir, meta.file),
                        parse_dates=["date"])
        x = d[args.col].to_numpy(dtype=float)
        dates = d["date"].to_numpy()

        for variant in args.variants:
            idx, truth, pred = forecast_trace(
                x, dates, args.T, args.horizon,
                fine_tune=(variant == "finetuned"),
                model_path=args.model_path)
            se = (truth - pred) ** 2
            mse, mase_v = float(se.mean()), mase(pred, truth)
            rows.append(dict(file=meta.file, alpha=meta.alpha,
                             seed=int(meta.seed), H_theory=meta.H_theory,
                             variant=variant, n_fc=len(truth),
                             mse=mse, mase=mase_v))
            print(f"{meta.file:>24} {variant:>10} {len(truth):>6d} "
                  f"{mse:>13.4e} {mase_v:>9.4f}")

            if args.per_fc_out:
                os.makedirs(args.per_fc_out, exist_ok=True)
                pd.DataFrame(dict(idx=idx, truth=truth, chronos=pred)).to_csv(
                    os.path.join(args.per_fc_out,
                                 f"fc_{variant}_{meta.file}"), index=False)

    df = pd.DataFrame(rows)
    df.insert(0, "run", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.insert(1, "T", args.T)
    df.insert(2, "horizon", args.horizon)
    df.insert(3, "model", args.model_path)

    lines = summary_lines(df, args)
    print("\n" + "\n".join(lines))

    if args.res_txt:
        append_res_txt(args.res_txt, df, args)
        print(f"per-forecaster MASE lines appended -> {args.res_txt}")
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        exists = os.path.exists(args.out) and os.path.getsize(args.out) > 0
        df.to_csv(args.out, mode="a", header=not exists, index=False)
        print(f"per-trace rows appended -> {args.out}")
    if args.summary_out:
        os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
        with open(args.summary_out, "a") as fh:
            fh.write("\n".join(lines) + "\n\n")
        print(f"summary appended -> {args.summary_out}")


if __name__ == "__main__":
    main()
