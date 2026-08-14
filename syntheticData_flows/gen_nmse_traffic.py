"""Generate the synthetic ON/OFF Pareto traces used for the NMSE forecasting study.

Defaults reproduce the original ./nmse_traffic set: 500 sources, 1 Mb/s while
ON, xm_on = xm_off = 1 ms, 150 ms bins, 1440 s horizon. 10 seeds x 3 tail
indices (alpha_on = alpha_off in {1.04, 1.24, 1.44}) = 30 traces, written in
the same (date, OT) layout as the other datasets.

    python gen_nmse_traffic.py
    python gen_nmse_traffic.py --n-sources 5000 --dt 1 --T 9600 \
        --outdir ./nmse_traffic_x5000_dt1s
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch

import gen_onoff_pareto as gp

# gen_onoff_pareto.estimate_hurst uses np but the module never imports it.
gp.np = np

RATE_ON = 1e6            # 1 Mb/s per ON source
XM_ON = 0.001            # minimum ON duration [s]
XM_OFF = 0.001           # minimum OFF duration [s]
STATIONARY_START = True  # steady state: no cold-start ramp to pollute forecasts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-sources", type=int, default=500)
    ap.add_argument("--dt", type=float, default=0.15, help="bin width [s]")
    ap.add_argument("--T", type=float, default=1440.0,
                    help="horizon kept after burn-in [s]; samples = T/dt")
    ap.add_argument("--burn-in", type=float, default=360.0,
                    help="[s] simulated then discarded, on top of T")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.04, 1.24, 1.44])
    ap.add_argument("--n-seeds", type=int, default=10,
                    help="seeds 0 .. n_seeds-1")
    ap.add_argument("--outdir", default="./nmse_traffic")
    args = ap.parse_args()

    N_SOURCES, DT, T, BURN_IN = args.n_sources, args.dt, args.T, args.burn_in
    ALPHAS, SEEDS, OUTDIR = args.alphas, list(range(args.n_seeds)), args.outdir

    os.makedirs(OUTDIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_burn = int(round(BURN_IN / DT))
    n_bins = int(round(T / DT))
    print(f"device={device}  X={N_SOURCES}  dt={DT}s  T={T}s  bins={n_bins}  "
          f"burn_in={BURN_IN}s ({n_burn} bins)  "
          f"traces={len(ALPHAS) * len(SEEDS)}")

    rows = []
    for alpha in ALPHAS:
        H_theory = (3.0 - alpha) / 2.0
        for seed in SEEDS:
            # Simulate T + BURN_IN and keep only the tail: any transient from
            # the t=0 initialisation has decayed by then, whichever start mode
            # is used.
            rate, p_on = gp.simulate_onoff_aggregate(
                N_SOURCES, T + BURN_IN, DT, alpha, alpha, XM_ON, XM_OFF,
                RATE_ON, device=device, seed=seed,
                stationary_start=STATIONARY_START)
            rate_np = rate[n_burn:n_burn + n_bins].cpu().numpy()
            del rate
            if device.type == "cuda":
                torch.cuda.empty_cache()

            H_hat = gp.estimate_hurst(
                rate_np, min_scale_bins=5 * max(XM_ON, XM_OFF) / DT)
            m_theory = N_SOURCES * RATE_ON * p_on

            # OT stores work *per bin* (rate * dt), matching f_sns100.csv;
            # downstream code recovers the rate as mean(OT)/dt.
            idx = pd.date_range("2025-01-01", periods=len(rate_np),
                                freq=f"{int(DT * 1000)}ms")
            name = f"onoff_a{alpha:.2f}_s{seed:02d}.csv"
            pd.DataFrame({"date": idx, "OT": rate_np * DT}).to_csv(
                os.path.join(OUTDIR, name), index=False)

            rows.append(dict(file=name, alpha=alpha, seed=seed, dt=DT, T=T,
                             n_sources=N_SOURCES, rate_on=RATE_ON,
                             xm_on=XM_ON, xm_off=XM_OFF,
                             stationary_start=STATIONARY_START,
                             burn_in=BURN_IN,
                             H_theory=H_theory, H_hat=H_hat,
                             mean_rate=float(rate_np.mean()),
                             mean_rate_theory=m_theory,
                             std_rate=float(rate_np.std())))
            print(f"{name}  H_hat={H_hat:.3f} (theory {H_theory:.3f})  "
                  f"mean={rate_np.mean():.3e} (theory {m_theory:.3e})")

    man = os.path.join(OUTDIR, "manifest.csv")
    pd.DataFrame(rows).to_csv(man, index=False)
    print(f"\nsaved {len(rows)} traces + {man}")
    summary = pd.DataFrame(rows).groupby("alpha")[["H_theory", "H_hat",
                                                   "mean_rate"]].mean()
    print(summary)


if __name__ == "__main__":
    main()
