# Three-cell virtual buffer simulation driven by independent Pareto ON/OFF
# traffic streams (see gen_onoff_pareto.py).

import numpy as np
import pandas as pd
import torch

from gen_onoff_pareto import local_whittle_H, simulate_onoff_aggregate


def analyze_traffic_model(rate_data, dt):
    increments = np.array(rate_data)

    H_hat = local_whittle_H(increments)

    m_hat = np.mean(increments) / dt
    a2_hat = np.std(increments) / (m_hat * dt**H_hat)
    a_hat = (a2_hat ** 2) * m_hat

    return m_hat, a_hat, H_hat


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

ARRIVAL_DT = 0.001      # traffic-generation granularity [s]
SERVICE_DT = 0.001      # buffer-service granularity [s]
T = 100.0               # recorded horizon [s]
WARMUP = 1.0            # extra leading seconds, generated then discarded [s]

# 2.75e8 b/s = 275,000 bits drained per 1 ms tick.
SERVICE_RATE = 2.75e8

# One entry per cell. Each cell has its own, independent set of ON/OFF sources.
CELLS = [
    # name,      n_sources, alpha_on, alpha_off, xm_on, xm_off, rate_on [b/s],
    #                                                   service rate [b/s], seed
    dict(name="cell0", n_sources=500, alpha_on=1.04, alpha_off=1.04,
         xm_on=0.001, xm_off=0.001, rate_on=1e6, service_rate=SERVICE_RATE,
         seed=101),
    dict(name="cell1", n_sources=500, alpha_on=1.24, alpha_off=1.24,
         xm_on=0.001, xm_off=0.001, rate_on=1e6, service_rate=SERVICE_RATE,
         seed=202),
    dict(name="cell2", n_sources=500, alpha_on=1.44, alpha_off=1.44,
         xm_on=0.001, xm_off=0.001, rate_on=1e6, service_rate=SERVICE_RATE,
         seed=303),
]

BUFFER_CAPACITY = None      # None = infinite; else [bits], excess is dropped

# Forecasting view on top of the raw ARRIVAL_DT demand: consecutive bins are
# summed into samples of FORECASTING_GRANULARITY bins each, and the last
# PAST_WINDOW samples are kept as the history fed to a predictor.
FORECASTING_GRANULARITY = 15    # raw bins per forecasting sample
PAST_WINDOW = 24                # samples retained -> span = G * PAST_WINDOW bins


# ----------------------------------------------------------------------------
# Traffic generation
# ----------------------------------------------------------------------------

def generate_demand(cell, T, dt, device, warmup=0.0):
    """Demand [bits] per arrival bin for one cell, shape (T/dt,).

    Generates warmup + T seconds and returns only the tail, so the cold-start
    transient of the ON/OFF sources never reaches the buffer.
    """
    rate, p_on = simulate_onoff_aggregate(
        cell["n_sources"], T + warmup, dt, cell["alpha_on"], cell["alpha_off"],
        cell["xm_on"], cell["xm_off"], cell["rate_on"],
        device=device, seed=cell["seed"], stationary_start=False)
    demand = rate.cpu().numpy() * dt   # rate is a per-bin average rate
    return demand[int(round(warmup / dt)):], p_on


# ----------------------------------------------------------------------------
# Forecasting view
# ----------------------------------------------------------------------------

def aggregate_demand(demand, granularity):
    """Sum every `granularity` consecutive raw bins into one sample.

    A trailing partial group is dropped, so the result covers
    len(demand) // granularity * granularity raw bins.
    """
    n = len(demand) // granularity * granularity
    return demand[:n].reshape(-1, granularity).sum(axis=1)


def build_past_windows(samples, past_window):
    """Sliding history of the aggregated demand.

    Row i is the `past_window` samples ending at sample index
    i + past_window - 1, i.e. the history a predictor would see when standing
    at that point in time. Shape (len(samples) - past_window + 1, past_window).
    """
    if len(samples) < past_window:
        raise ValueError(f"need at least {past_window} samples, "
                         f"got {len(samples)}")
    return np.lib.stride_tricks.sliding_window_view(
        samples, past_window).copy()


# ----------------------------------------------------------------------------
# Buffer dynamics
# ----------------------------------------------------------------------------

def run_buffers(demand, service_rate, arrival_dt, service_dt, capacity=None):
    """Drive one buffer at the service tick resolution.

    Returns (occupancy, served, dropped) sampled at every service tick, all in
    bits. Occupancy is recorded *after* the tick's arrival and service.
    """
    ticks_per_bin = int(round(arrival_dt / service_dt))
    n_ticks = len(demand) * ticks_per_bin
    served_per_tick = service_rate * service_dt

    occupancy = np.empty(n_ticks, dtype=np.float64)
    served = np.empty(n_ticks, dtype=np.float64)
    dropped = np.zeros(n_ticks, dtype=np.float64)

    q = 0.0
    for k in range(n_ticks):
        if k % ticks_per_bin == 0:            # arrival instant
            q += demand[k // ticks_per_bin]
            if capacity is not None and q > capacity:
                dropped[k] = q - capacity
                q = capacity
        s = min(q, served_per_tick)           # service instant
        q -= s
        served[k] = s
        occupancy[k] = q
    return occupancy, served, dropped


# ----------------------------------------------------------------------------

def seed_sweep(seeds, device):
    """Per-cell H_est and one-step NMSE across independent traffic realizations.

    Skips the buffer simulation: only the generated demand is needed. Each cell
    keeps its own alpha (hence its own theoretical H) but is re-seeded, so the
    spread across a row is realization noise, not a change of regime.
    """
    import scipy.linalg
    from scipy import integrate
    from norrosForecast_original import g_T

    T_taps = PAST_WINDOW
    sample_dt = ARRIVAL_DT * FORECASTING_GRANULARITY

    def acov(x, lag):
        if lag == 0:
            return np.var(x)
        mu = np.mean(x)
        return np.mean((x[:-lag] - mu) * (x[lag:] - mu))

    def one_run(samples):
        n = len(samples)
        train, test = samples[:int(n * 0.7)], samples[int(n * 0.8):]
        m, a, H = analyze_traffic_model(train, sample_dt)
        Z_tr = (train - m * sample_dt) / np.sqrt(m * a)
        Z_te = (test - m * sample_dt) / np.sqrt(m * a)

        row = np.array([acov(Z_tr, k) for k in range(T_taps)])
        w_yw = np.linalg.solve(
            scipy.linalg.toeplitz(row) + 1e-8 * np.eye(T_taps),
            np.array([acov(Z_tr, 1 + k) for k in range(T_taps)]))
        w_g = np.array([integrate.quad(lambda t: g_T(1.0, t, float(T_taps), H),
                                       k, k + 1, limit=200)[0]
                        for k in range(T_taps)])

        win = np.lib.stride_tricks.sliding_window_view(Z_te, T_taps)[:-1][:, ::-1]
        truth = Z_te[T_taps:]
        nmse = lambda w: np.mean((truth - win @ w) ** 2) / np.var(truth)
        return H, nmse(w_yw), nmse(w_g), nmse(np.eye(T_taps)[0])

    print(f"seed sweep: {len(seeds)} seeds x {len(CELLS)} cells   "
          f"T={T:g}s  granularity={FORECASTING_GRANULARITY} "
          f"({sample_dt*1e3:.0f}ms)  past_window={PAST_WINDOW}")
    print("H via local Whittle\n")

    results = {}
    for cell in CELLS:
        H_theory = (3.0 - min(cell["alpha_on"], cell["alpha_off"])) / 2.0
        print(f"{cell['name']}  alpha={cell['alpha_on']}  "
              f"X={cell['n_sources']}  H_theory={H_theory:.3f}")
        print(f"{'seed':>8} {'H_est':>8} | {'YW':>8} {'g_T':>8} {'naive':>8} "
              f"| {'gap%':>7}")
        rows = []
        for seed in seeds:
            rate, _ = simulate_onoff_aggregate(
                cell["n_sources"], T + WARMUP, ARRIVAL_DT,
                cell["alpha_on"], cell["alpha_off"], cell["xm_on"],
                cell["xm_off"], cell["rate_on"], device=device, seed=seed,
                stationary_start=False)
            demand = rate.cpu().numpy() * ARRIVAL_DT
            del rate
            demand = demand[int(round(WARMUP / ARRIVAL_DT)):]
            samples = aggregate_demand(demand, FORECASTING_GRANULARITY)

            H, yw, gt, nv = one_run(samples)
            gap = 100.0 * (gt - yw) / yw
            rows.append((H, yw, gt, nv, gap))
            print(f"{seed:>8} {H:>8.3f} | {yw:>8.4f} {gt:>8.4f} {nv:>8.4f} "
                  f"| {gap:>6.1f}%")

        arr = np.array(rows)
        mean, sd = arr.mean(axis=0), arr.std(axis=0, ddof=1)
        print(f"{'mean':>8} {mean[0]:>8.3f} | {mean[1]:>8.4f} {mean[2]:>8.4f} "
              f"{mean[3]:>8.4f} | {mean[4]:>6.1f}%")
        print(f"{'sd':>8} {sd[0]:>8.3f} | {sd[1]:>8.4f} {sd[2]:>8.4f} "
              f"{sd[3]:>8.4f} | {sd[4]:>6.1f}%")

        # paired comparison: both predictors see the identical realization
        from scipy import stats
        d = arr[:, 2] - arr[:, 1]
        n = len(d)
        half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        t_stat, p = stats.ttest_rel(arr[:, 2], arr[:, 1])
        print(f"  paired (g_T - YW): {d.mean():+.4f} +/- {half:.4f}   "
              f"t={t_stat:.2f}  p={p:.4f}   g_T better in {int((d < 0).sum())}/{n}\n")
        results[cell["name"]] = arr
    return results


if __name__ == "__main__":
    import sys

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "--sweep" in sys.argv:
        seed_sweep([101, 202, 303, 404, 505, 606, 707, 808, 909, 1010], device)
        sys.exit(0)

    ticks_per_bin = int(round(ARRIVAL_DT / SERVICE_DT))
    n_bins = int(round(T / ARRIVAL_DT))
    sample_dt = ARRIVAL_DT * FORECASTING_GRANULARITY
    n_samples = n_bins // FORECASTING_GRANULARITY
    print(f"device={device}  T={T}s (+{WARMUP}s warmup, discarded)  "
          f"arrival_dt={ARRIVAL_DT*1e3:.0f}ms ({n_bins} bins)  "
          f"service_dt={SERVICE_DT*1e3:.0f}ms "
          f"({n_bins * ticks_per_bin} ticks)")
    print(f"forecasting: granularity={FORECASTING_GRANULARITY} bins "
          f"({sample_dt*1e3:.0f}ms) -> {n_samples} samples   "
          f"past_window={PAST_WINDOW} -> span {PAST_WINDOW*sample_dt*1e3:.0f}ms")

    out = {}
    raw = {}
    agg = {}
    windows = {}
    for cell in CELLS:
        demand, p_on = generate_demand(cell, T, ARRIVAL_DT, device, WARMUP)
        occ, served, dropped = run_buffers(
            demand, cell["service_rate"], ARRIVAL_DT, SERVICE_DT,
            BUFFER_CAPACITY)

        arrival_rate = demand.sum() / T
        util = arrival_rate / cell["service_rate"]
        print(f"\n{cell['name']}: X={cell['n_sources']} alpha={cell['alpha_on']} "
              f"p_on={p_on:.3f}")
        print(f"  offered   {arrival_rate:.3e} b/s   "
              f"service {cell['service_rate']:.3e} b/s   rho={util:.3f}")
        print(f"  buffer    mean {occ.mean():.3e}  p99 "
              f"{np.percentile(occ, 99):.3e}  max {occ.max():.3e} bits")
        print(f"  empty for {100.0 * (occ <= 0).mean():.1f}% of ticks   "
              f"dropped {dropped.sum():.3e} bits")

        # Norros fBm fit on the raw generated demand (all T/ARRIVAL_DT bins).
        m_hat, a_hat, H_hat = analyze_traffic_model(demand, ARRIVAL_DT)
        H_theory = (3.0 - min(cell["alpha_on"], cell["alpha_off"])) / 2.0
        print(f"  fBm fit   m={m_hat:.4e} b/s   a={a_hat:.4e} b   "
              f"H={H_hat:.3f} (theory {H_theory:.3f})")

        samples = aggregate_demand(demand, FORECASTING_GRANULARITY)
        win = build_past_windows(samples, PAST_WINDOW)
        print(f"  samples   {samples.shape}  mean {samples.mean():.3e} bits "
              f"per {sample_dt*1e3:.0f}ms   windows {win.shape}")

        out[f"{cell['name']}_demand"] = np.repeat(demand / ticks_per_bin,
                                                  ticks_per_bin)
        out[f"{cell['name']}_buffer"] = occ
        out[f"{cell['name']}_served"] = served
        raw[cell["name"]] = demand
        agg[cell["name"]] = samples
        windows[cell["name"]] = win

    idx = pd.date_range("2025-01-01", periods=n_bins * ticks_per_bin,
                        freq=f"{int(SERVICE_DT * 1000)}ms")
    df = pd.DataFrame({"date": idx, **out})
    df.to_csv("./cell_buffer_sim.csv", index=False)
    print("\nsaved ./cell_buffer_sim.csv")

    # Aggregated demand at the forecasting granularity, plus the sliding
    # history windows (one array per cell, shape (n_windows, PAST_WINDOW)).
    agg_idx = pd.date_range("2025-01-01", periods=n_samples,
                            freq=f"{int(sample_dt * 1000)}ms")
    pd.DataFrame({"date": agg_idx, **agg}).to_csv(
        "./cell_demand_agg.csv", index=False)
    np.savez("./cell_demand_windows.npz",
             granularity=FORECASTING_GRANULARITY, past_window=PAST_WINDOW,
             sample_dt=sample_dt, **windows)
    print("saved ./cell_demand_agg.csv and ./cell_demand_windows.npz")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.arange(n_bins * ticks_per_bin) * SERVICE_DT
    fig, axes = plt.subplots(len(CELLS), 1, figsize=(11, 8), sharex=True)
    for ax, cell in zip(axes, CELLS):
        ax.plot(t, out[f"{cell['name']}_buffer"], lw=0.5)
        ax.set_ylabel(f"{cell['name']}\nbuffer [bits]")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle("Per-cell virtual buffer occupancy (1 ms service ticks)")
    fig.tight_layout()
    fig.savefig("cell_buffer_sim.png", dpi=120)
    print("saved cell_buffer_sim.png")

    # Generated demand at the raw ARRIVAL_DT granularity. Full horizon on the
    # left, a short zoom on the right where individual bins are resolvable.
    t_bin = np.arange(n_bins) * ARRIVAL_DT
    zoom_s = 1.0
    n_zoom = int(round(zoom_s / ARRIVAL_DT))
    fig, axes = plt.subplots(len(CELLS), 2, figsize=(13, 8), sharex="col")
    for (ax_full, ax_zoom), cell in zip(axes, CELLS):
        d = raw[cell["name"]] / 1e6                      # bits -> Mbit per bin
        ax_full.plot(t_bin, d, lw=0.3)
        ax_full.axhline(cell["service_rate"] * ARRIVAL_DT / 1e6,
                        color="r", lw=0.8, ls="--")
        ax_full.set_ylabel(f"{cell['name']}\nMbit / bin")
        ax_zoom.plot(t_bin[:n_zoom], d[:n_zoom], lw=0.7)
        ax_zoom.axhline(cell["service_rate"] * ARRIVAL_DT / 1e6,
                        color="r", lw=0.8, ls="--", label="service capacity")
        ax_zoom.set_title(f"alpha={cell['alpha_on']}", fontsize=9)
    axes[0][1].legend(fontsize=8)
    axes[-1][0].set_xlabel("time [s]")
    axes[-1][1].set_xlabel(f"time [s]  (first {zoom_s:g}s)")
    fig.suptitle(f"Generated demand per {ARRIVAL_DT*1e3:.0f}ms bin "
                 f"(full horizon | zoom)")
    fig.tight_layout()
    fig.savefig("cell_demand.png", dpi=120)
    print("saved cell_demand.png")

    # Same traffic seen at the forecasting granularity.
    t_smp = np.arange(n_samples) * sample_dt
    fig, axes = plt.subplots(len(CELLS), 1, figsize=(11, 7), sharex=True)
    for ax, cell in zip(axes, CELLS):
        ax.plot(t_smp[:int(round(10.0 / sample_dt))],
                agg[cell["name"]][:int(round(10.0 / sample_dt))] / 1e6, lw=0.7)
        ax.set_ylabel(f"{cell['name']}\nMbit / sample")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Demand aggregated to {sample_dt*1e3:.0f}ms samples "
                 f"(first 10s)")
    fig.tight_layout()
    fig.savefig("cell_demand_agg.png", dpi=120)
    print("saved cell_demand_agg.png")
