# Generate the single-cell ON/OFF traffic once and keep it on disk.
# python make_traffic.py --sample-dt 1 --minutes 15 --alpha 1.04 --n-seeds 10


import argparse
import os

import numpy as np
import pandas as pd
import torch

from gen_onoff_pareto import simulate_onoff_aggregate

ARRIVAL_DT = 0.001          # raw generator granularity [s]
START = "2025-01-01 00:00:00"
DATA_DIR = "./datasets/"


def freq_str(sample_dt):
    """The offset alias for a sample interval of `sample_dt` seconds.

    Whole seconds go out as seconds ('1s', '90s'), anything else as whole
    milliseconds ('100L', '15L', '1500L').

    Any whole-millisecond interval is allowed: utils/timefeatures.py resolves
    sub-second frequencies generically and models/embed.py takes its input
    width from that same list, so the two cannot disagree.
    """
    ms = sample_dt * 1000.0
    if abs(ms - round(ms)) > 1e-9:
        raise ValueError(f"sample_dt={sample_dt} is not a whole number of ms")
    ms = int(round(ms))
    if ms <= 0:
        raise ValueError(f"sample_dt={sample_dt} must be positive")
    if ms % 1000 == 0:
        return f"{ms // 1000}s"
    return f"{ms}L"


def dataset_name(sample_dt, minutes, alpha, index):
    """The filename one dataset lives under. The single source of truth."""
    return (f"onoff_{freq_str(sample_dt)}_{minutes:g}min"
            f"_a{alpha:g}_s{index:02d}.csv")


def dataset_path(sample_dt, minutes, alpha, index, data_dir=DATA_DIR):
    return os.path.join(data_dir, dataset_name(sample_dt, minutes, alpha, index))


def seed_of(index):
    """Generator seed for seed index 0,1,2,... -- 101, 202, 303, ..."""
    return 101 * (index + 1)


def generate(index, sample_dt, minutes, alpha, n_sources, rate_on, device):

    seconds = minutes * 60.0
    rate, _ = simulate_onoff_aggregate(
        n_sources, seconds, sample_dt, alpha, alpha,
        xm_on=0.001, xm_off=0.001, rate_on=rate_on,
        device=device, seed=seed_of(index), stationary_start=True)
    demand = rate.cpu().numpy() * sample_dt
    del rate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return demand


def write_csv(samples, path, sample_dt):
    dates = pd.date_range(START, periods=len(samples), freq=freq_str(sample_dt))
    # %.17g so the file holds the exact float64 the generator produced rather
    # than pandas' shorter repr.
    pd.DataFrame({"date": dates, "OT": samples}).to_csv(
        path, index=False, float_format="%.17g")


def read_csv(path):
    """The OT column: the aggregated demand samples.

    float_precision='round_trip' because the default C parser is not correctly
    rounded and lands a ulp off on about half the values.
    """
    df = pd.read_csv(path, float_precision="round_trip")
    return df["OT"].values.astype(float)


def load(index, sample_dt, minutes, alpha, data_dir=DATA_DIR):
    """Read one dataset by its parameters. Raises if it was never generated."""
    path = dataset_path(sample_dt, minutes, alpha, index, data_dir)
    if not os.path.exists(path):
        raise SystemExit(f"no dataset at {path}\n"
                         f"generate it with: python make_traffic.py "
                         f"--sample-dt {sample_dt:g} --minutes {minutes:g} "
                         f"--alpha {alpha:g} --n-seeds {index + 1}")
    return read_csv(path)


def add_dataset_args(ap):
    """The four parameters that identify a dataset. Shared by every consumer."""
    ap.add_argument("--sample-dt", type=float, default=1.0,
                    help="forecasting sample granularity [s]")
    ap.add_argument("--minutes", type=float, default=15.0,
                    help="length of each realization [min]")
    ap.add_argument("--alpha", type=float, default=1.04,
                    help="Pareto tail index for both ON and OFF; H=(3-alpha)/2")
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="use seed indices 0 .. n_seeds-1")
    ap.add_argument("--data-dir", default=DATA_DIR)
    return ap


def main():
    ap = add_dataset_args(argparse.ArgumentParser())
    ap.add_argument("--n-sources", type=int, default=1500)
    ap.add_argument("--rate-on", type=float, default=1e6,
                    help="bits/s per ON source")
    ap.add_argument("--force", action="store_true",
                    help="regenerate datasets whose CSV already exists")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_theory = (3.0 - args.alpha) / 2.0
    n_expect = int(round(args.minutes * 60.0 / args.sample_dt))
    print(f"device={device}  X={args.n_sources}  alpha={args.alpha} "
          f"(H_theory={H_theory:.3f})  rate_on={args.rate_on:g}")
    print(f"{args.minutes:g} min, ON/OFF quantum {ARRIVAL_DT*1e3:.0f}ms "
          f"-> {n_expect} samples of {args.sample_dt:g}s\n")

    os.makedirs(args.data_dir, exist_ok=True)
    for i in range(args.n_seeds):
        path = dataset_path(args.sample_dt, args.minutes, args.alpha, i,
                            args.data_dir)
        if os.path.exists(path) and not args.force:
            print(f"  s{i:02d}  exists, skipped   {path}")
            continue
        s = generate(i, args.sample_dt, args.minutes, args.alpha,
                     args.n_sources, args.rate_on, device)
        write_csv(s, path, args.sample_dt)
        print(f"  s{i:02d}  seed={seed_of(i):<5} {len(s):>6} samples -> {path}")

    print(f"\nn_sources and rate_on are not in the filename: if you change "
          f"them, use a fresh --data-dir or --force.")


if __name__ == "__main__":
    main()
