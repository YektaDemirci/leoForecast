"""DLinear on the synthetic ON/OFF traces: one model per trace, 1 step ahead.

Outputs (all appended, so repeated runs accumulate):
  results/mase_traffic_dl_summary.txt   per-alpha MSE + MASE, mean +- CI
  results/mase_traffic_dl.csv           one row per trace
  results/mase_traffic_res.txt          long-format scoreboard, shared with
                                        the analytic and Chronos runs
"""

import argparse
import datetime
import os
import random

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch.utils.data import DataLoader

from data_provider.data_factory import data_provider
from exp.exp_main import Exp_Main

DATA_DIR = "./nmse_traffic"
FIX_SEED = 2021


def ci(x, conf=0.95):
    """Mean and t-based CI half-width. Each seed is one independent replicate."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return x.mean(), float("nan")
    return x.mean(), stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def build_args(a, data_path):
    """run_longExp.py's argument namespace, for one trace.

    Only the fields listed in the module docstring depart from that script's
    defaults; everything else is copied verbatim so a DLinear run here is the
    same DLinear run the shell scripts launch.
    """
    ns = argparse.Namespace(
        is_training=1, model_id=os.path.splitext(data_path)[0], model="DLinear",
        # data
        data="custom", root_path=a.data_dir, data_path=data_path,
        features="S", target=a.col, freq="S", checkpoints=a.checkpoints,
        # forecasting task
        seq_len=a.seq_len, label_len=a.label_len, pred_len=a.pred_len,
        # model
        individual=False, embed_type=0, enc_in=1, dec_in=1, c_out=1,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, d_ff=2048,
        moving_avg=25, factor=1, distil=True, dropout=0.05, embed="timeF",
        activation="gelu", output_attention=False, do_predict=False,
        # optimisation
        num_workers=a.num_workers, itr=1, train_epochs=a.train_epochs,
        batch_size=a.batch_size, patience=a.patience,
        learning_rate=a.learning_rate, des="test", loss="mse", lradj="type1",
        use_amp=False, parr="nmse_traffic",
        # gpu
        use_gpu=torch.cuda.is_available() and a.use_gpu, gpu=a.gpu,
        use_multi_gpu=False, devices="0", test_flop=False,
    )
    # run_longExp.py keeps the verbose string on detail_freq and hands the
    # single-letter code to the time-feature encoder.
    ns.detail_freq = ns.freq
    ns.freq = ns.freq[-1:]
    return ns


def forecast_trace(a, meta):
    """Train one DLinear on one trace, then predict its whole test block.

    Returns (idx, truth, pred) in raw traffic units, with idx the GLOBAL sample
    index of each forecast target -- the same keying the analytic per-forecast
    dumps use, so the two join bucket by bucket.
    """
    args = build_args(a, meta.file)
    setting = "{}_{}_{}_{}_{}_{}".format(args.data_path, args.model, args.parr,
                                         args.seq_len, args.pred_len,
                                         args.detail_freq)

    # Same seeding as run_longExp.py: identical init for every trace, so
    # across-seed spread is the trace's, not the initialiser's.
    random.seed(FIX_SEED)
    torch.manual_seed(FIX_SEED)
    np.random.seed(FIX_SEED)

    exp = Exp_Main(args)
    exp.train(setting)                       # early stopping restores the best

    # Our own pass over the test set: drop_last=False keeps the tail windows
    # data_provider's test loader would discard.
    test_data, _ = data_provider(args, flag="test")
    loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, drop_last=False)

    exp.model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch_x, batch_y, _, _ in loader:
            out = exp.model(batch_x.float().to(exp.device))
            preds.append(out[:, -args.pred_len:, 0].cpu().numpy())
            trues.append(batch_y[:, -args.pred_len:, 0].numpy())
    pred = np.concatenate(preds)             # (N, pred_len), standardised
    truth = np.concatenate(trues)

    # Back to traffic units. features='S' means the scaler holds one column, so
    # the flattened series can be pushed through it directly.
    sd = float(np.sqrt(test_data.scaler.var_[0]))
    mu = float(test_data.scaler.mean_[0])
    pred, truth = pred * sd + mu, truth * sd + mu

    # Dataset_Custom's test block begins seq_len samples before the split, so
    # window i's target block starts at test_start + i.
    n = len(pd.read_csv(os.path.join(args.root_path, args.data_path)))
    test_start = n - int(n * 0.2)
    idx = test_start + np.arange(len(pred))

    if a.align == "analytic":
        keep = args.seq_len                  # first seq_len targets are history
        idx, pred, truth = idx[keep:], pred[keep:], truth[keep:]

    torch.cuda.empty_cache()
    # pred_len > 1 is scored on the CUMULATIVE demand over the horizon, as the
    # analytic predictors are, so the two measure the same quantity.
    return idx, truth.sum(axis=1), pred.sum(axis=1)


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
    """One line per trace, in the long format score_nmse_traffic.append_res_txt
    writes, so every forecaster's scoreboard concatenates."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a") as fh:
        if new:
            fh.write(RES_TXT_HDR + "\n")
        for _, r in df.iterrows():
            fh.write(f"{'DLINEAR':<14} {args.seq_len:>8d} {r.file:>24} "
                     f"{r.alpha:>6.2f} {int(r.seed):>5d} "
                     f"{r.mse:>13.4e} {r.mase:>9.4f}\n")


def summary_lines(df, args):
    """Per-alpha mean +- 95% t-CI half-width over seeds, for MSE and MASE."""
    L = [f"=== {df['run'].iloc[0]}  DLINEAR  seq_len={args.seq_len} "
         f"label_len={args.label_len} pred_len={args.pred_len} "
         f"align={args.align}  {args.data_dir} ===",
         "mean over seeds +- half-width of the 95% t-CI over seeds "
         "(seeds are the unit of analysis)"]
    for alpha, g in df.groupby("alpha"):
        L.append(f"\nalpha={alpha:.2f}  H_theory={g.H_theory.iloc[0]:.2f}  "
                 f"n_seeds={g.seed.nunique()}  n_fc={int(g.n_fc.iloc[0])}")
        mm, mh = ci(g.mse)
        nm, nh = ci(g.mase)
        L.append(f"{'MSE':>10} {mm:>13.4e} +- {mh:.2e}")
        L.append(f"{'MASE':>10} {nm:>13.4f} +- {nh:.4f}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="directory holding the trace CSVs and manifest.csv")
    ap.add_argument("--col", default="OT")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--label_len", type=int, default=24)
    ap.add_argument("--pred_len", type=int, default=1)
    ap.add_argument("--align", choices=["analytic", "native"],
                    default="analytic",
                    help="analytic: first target at test_start + seq_len, as "
                         "the analytic and Chronos runs; native: keep every "
                         "window Dataset_Custom yields")
    ap.add_argument("--train_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.0001)
    ap.add_argument("--num_workers", type=int, default=10)
    ap.add_argument("--checkpoints", default="./checkpoints/")
    ap.add_argument("--use_gpu", type=bool, default=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--per-fc-out", default=None,
                    help="directory for per-forecast dumps (one CSV per "
                         "trace); skipped if not given")
    ap.add_argument("--out", default=os.path.join("results",
                                                  "mase_traffic_dl.csv"))
    ap.add_argument("--res-txt", default=os.path.join("results",
                                                      "mase_traffic_res.txt"))
    ap.add_argument("--summary-out",
                    default=os.path.join("results",
                                         "mase_traffic_dl_summary.txt"))
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))
    if args.alphas:
        man = man[man.alpha.isin(args.alphas)]
    if args.seeds is not None:
        man = man[man.seed.isin(args.seeds)]

    print(f"{len(man)} traces from {args.data_dir}  DLinear  "
          f"seq_len={args.seq_len} label_len={args.label_len} "
          f"pred_len={args.pred_len}  align={args.align}")
    hdr = f"{'file':>24} {'n_fc':>6} {'MSE':>13} {'MASE':>9}"

    rows = []
    for _, meta in man.iterrows():
        idx, truth, pred = forecast_trace(args, meta)
        se = (truth - pred) ** 2
        mse, mase_v = float(se.mean()), mase(pred, truth)
        rows.append(dict(file=meta.file, alpha=meta.alpha, seed=int(meta.seed),
                         H_theory=meta.H_theory, n_fc=len(truth),
                         mse=mse, mase=mase_v))
        print("\n" + hdr + "\n" + "-" * len(hdr))
        print(f"{meta.file:>24} {len(truth):>6d} {mse:>13.4e} {mase_v:>9.4f}\n")

        if args.per_fc_out:
            os.makedirs(args.per_fc_out, exist_ok=True)
            pd.DataFrame(dict(idx=idx, truth=truth, dlinear=pred)).to_csv(
                os.path.join(args.per_fc_out, f"fc_dlinear_{meta.file}"),
                index=False)

    df = pd.DataFrame(rows)
    df.insert(0, "run", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.insert(1, "seq_len", args.seq_len)
    df.insert(2, "label_len", args.label_len)
    df.insert(3, "pred_len", args.pred_len)
    df.insert(4, "model", "DLinear")

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
