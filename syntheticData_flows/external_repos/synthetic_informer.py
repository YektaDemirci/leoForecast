"""Informer on the synthetic ON/OFF traces, one model per trace.
Outputs (all appended, so repeated runs accumulate):
  results/mase_traffic_informer_summary.txt  per-alpha MSE + MASE, mean +- CI
  results/mase_traffic_informer.csv          one row per trace
  results/mase_traffic_res.txt               long-format scoreboard, shared
                                             with the analytic run
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

from data.data_loader import Dataset_Custom
from exp.exp_informer import Exp_Informer

DATA_DIR = "./nmse_traffic"
FREQ = "s"                  # 1 s buckets
SEQ_LEN = 48
LABEL_LEN = 24
PRED_LEN = 1
TEST_FRAC = 0.20            # must match Dataset_Custom's split


def ci(x, conf=0.95):
    """Mean and t-based CI half-width. Each seed is one independent replicate."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    return x.mean(), stats.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)


def build_args(args, data_path):
    """The subset of main_informer.py's namespace that Exp_Informer reads."""
    return argparse.Namespace(
        model='informer', data='custom', root_path=args.data_dir,
        data_path=data_path, features='S', target=args.col,
        freq=FREQ, detail_freq=FREQ, checkpoints=args.checkpoints,
        seq_len=args.seq_len, label_len=args.label_len, pred_len=PRED_LEN,
        enc_in=1, dec_in=1, c_out=1,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, s_layers=[3, 2, 1],
        d_ff=2048, factor=5, padding=0, distil=True, dropout=0.05,
        attn='prob', embed='timeF', activation='gelu', output_attention=False,
        do_predict=False, mix=True, cols=None,
        num_workers=0, itr=1, train_epochs=args.epochs, batch_size=32,
        patience=3, learning_rate=1e-4, des='Exp', loss='mse', lradj='type1',
        use_amp=False, inverse=False,
        use_gpu=torch.cuda.is_available(), gpu=0, use_multi_gpu=False,
        devices='0',
    )


def forecast_trace(args, data_path):
    """Train (unless --skip-train) one Informer on this trace and return its
    1-step test forecasts in traffic units, keyed by global sample index.

    Sample i of Dataset_Custom's test split targets absolute row
    border1 + i + seq_len, with border1 = n - num_test - seq_len, i.e. row
    n - num_test + i. The first `seq_len` of those are dropped to start where
    the analytic run starts.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    exp_args = build_args(args, data_path)
    setting = 'nmse_traffic_{}_{}_{}_{}'.format(
        data_path, args.seq_len, PRED_LEN, FREQ)
    exp = Exp_Informer(exp_args)

    ckpt = os.path.join(exp_args.checkpoints, setting, 'checkpoint.pth')
    if args.skip_train and os.path.exists(ckpt):
        print(f'  reusing {ckpt}')
    else:
        exp.train(setting)
    exp.model.load_state_dict(torch.load(ckpt))
    exp.model.eval()

    ds = Dataset_Custom(root_path=args.data_dir, flag='test',
                        size=[args.seq_len, args.label_len, PRED_LEN],
                        features='S', data_path=data_path, target=args.col,
                        scale=True, inverse=False, timeenc=1, freq=FREQ)
    loader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)

    preds, trues = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            out, y = exp._process_one_batch(ds, bx, by, bxm, bym)
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    # (N, pred_len, 1) -> (N,), pred_len == 1
    pred = np.concatenate(preds)[:, -1, 0]
    true = np.concatenate(trues)[:, -1, 0]
    inv = lambda v: ds.scaler.inverse_transform(v.reshape(-1, 1)).ravel()
    pred, true = inv(pred), inv(true)

    n = len(pd.read_csv(os.path.join(args.data_dir, data_path)))
    idx = (n - int(n * TEST_FRAC)) + np.arange(len(pred))
    # Align with score_nmse_traffic.py / synthetic_chronos.py: their first
    # target sits seq_len samples into the test block.
    keep = idx >= (n - int(n * TEST_FRAC)) + args.seq_len
    return idx[keep], true[keep], pred[keep]


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
    """One line per trace, same long format as score_nmse_traffic.append_res_txt
    so both scoreboards concatenate."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new = not (os.path.exists(path) and os.path.getsize(path) > 0)
    with open(path, "a") as fh:
        if new:
            fh.write(RES_TXT_HDR + "\n")
        for _, r in df.iterrows():
            fh.write(f"{'INFORMER':<14} {args.seq_len:>8d} "
                     f"{r.file:>24} {r.alpha:>6.2f} {int(r.seed):>5d} "
                     f"{r.mse:>13.4e} {r.mase:>9.4f}\n")


def summary_lines(df, args):
    """Per-alpha mean +- 95% t-CI half-width over seeds, MSE and MASE."""
    L = [f"=== {df['run'].iloc[0]}  INFORMER seq_len={args.seq_len} "
         f"label_len={args.label_len} pred_len={PRED_LEN} "
         f"epochs={args.epochs}  {args.data_dir} ===",
         "mean over seeds +- half-width of the 95% t-CI over seeds "
         "(seeds are the unit of analysis)"]
    for alpha, g in df.groupby("alpha"):
        mm, mh = ci(g.mse)
        nm, nh = ci(g.mase)
        L.append(f"\nalpha={alpha:.2f}  H_theory={g.H_theory.iloc[0]:.2f}  "
                 f"n_seeds={g.seed.nunique()}  n_fc={int(g.n_fc.iloc[0])}")
        L.append(f"{'MSE':>24} {'MASE':>20}")
        L.append(f"{mm:>13.4e} +- {mh:<8.2e} {nm:>9.4f} +- {nh:.4f}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=DATA_DIR,
                    help="directory holding the trace CSVs and manifest.csv")
    ap.add_argument("--col", default="OT")
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    ap.add_argument("--label-len", type=int, default=LABEL_LEN)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2021, help="torch/numpy seed")
    ap.add_argument("--checkpoints", default="./checkpoints/")
    ap.add_argument("--skip-train", action="store_true",
                    help="reuse an existing checkpoint when there is one")
    ap.add_argument("--alphas", type=float, nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--per-fc-out", default=None,
                    help="directory for per-forecast dumps (one CSV per trace); "
                         "skipped if not given")
    ap.add_argument("--out",
                    default=os.path.join("results", "mase_traffic_informer.csv"))
    ap.add_argument("--res-txt",
                    default=os.path.join("results", "mase_traffic_res.txt"),
                    help="appended in the same long format the analytic run "
                         "uses, so one file holds every forecaster")
    ap.add_argument("--summary-out",
                    default=os.path.join("results",
                                         "mase_traffic_informer_summary.txt"))
    args = ap.parse_args()

    man = pd.read_csv(os.path.join(args.data_dir, "manifest.csv"))
    if args.alphas:
        man = man[man.alpha.isin(args.alphas)]
    if args.seeds is not None:
        man = man[man.seed.isin(args.seeds)]

    print(f"{len(man)} traces from {args.data_dir}  INFORMER "
          f"seq_len={args.seq_len} label_len={args.label_len} "
          f"pred_len={PRED_LEN} epochs={args.epochs}")
    hdr = f"{'file':>24} {'n_fc':>6} {'MSE':>13} {'MASE':>9}"

    rows = []
    for _, meta in man.iterrows():
        print(f"\n[{meta.file}] alpha={meta.alpha} seed={int(meta.seed)}")
        idx, truth, pred = forecast_trace(args, meta.file)
        se = (truth - pred) ** 2
        mse, mase_v = float(se.mean()), mase(pred, truth)
        rows.append(dict(file=meta.file, alpha=meta.alpha,
                         seed=int(meta.seed), H_theory=meta.H_theory,
                         n_fc=len(truth), mse=mse, mase=mase_v))
        print(hdr)
        print(f"{meta.file:>24} {len(truth):>6d} {mse:>13.4e} {mase_v:>9.4f}")

        if args.per_fc_out:
            os.makedirs(args.per_fc_out, exist_ok=True)
            pd.DataFrame(dict(idx=idx, truth=truth, informer=pred)).to_csv(
                os.path.join(args.per_fc_out, f"fc_informer_{meta.file}"),
                index=False)

    df = pd.DataFrame(rows)
    df.insert(0, "run", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    df.insert(1, "seq_len", args.seq_len)
    df.insert(2, "label_len", args.label_len)
    df.insert(3, "pred_len", PRED_LEN)

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
