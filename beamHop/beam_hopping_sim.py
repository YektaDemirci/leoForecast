
#   python beam_hopping_sim.py
#   python beam_hopping_sim.py --sample-dt 0.015 --T 200 --alpha 1.04
#   python beam_hopping_sim.py --planners equal naive oracle   # skip R/ARFIMA
#   python beam_hopping_sim.py --backlog-aware                 # add queue state

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

# The generator and the Whittle estimator live one level up, in Informer/.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell_buffer_sim import analyze_traffic_model      # local-Whittle m, a, H
from gen_onoff_pareto import simulate_onoff_aggregate

ARRIVAL_DT = 0.001          # generation and beam granularity [s]
BEAM_RATE = 270e6           # bits/s drained while the beam is on a cell
WARMUP = 1.0                # generated then discarded [s]

# Per-cell virtual buffer capacity. Memory sizes are MiB (1024^2 bytes), the
# unit the hardware is actually specified in.
#
#   2 MiB = 16.78 Mbit = 62 ms of a full 270 Mb/s beam
#
BUFFER_MB = 2.0
MIB = 1024.0 ** 2


# ----------------------------------------------------------------------------
# Traffic
# ----------------------------------------------------------------------------

def generate_cells(n_cells, T, alpha, n_sources, rate_on, seeds, device):
    """Demand [bits] per 1 ms bin, shape (n_cells, T/ARRIVAL_DT).

    Identical parameters in every cell -- only the seed differs -- so the cells
    are statistically exchangeable and any asymmetry the planner exploits is a
    genuine short-term imbalance, not a permanent one.
    """
    out = []
    for seed in seeds[:n_cells]:
        rate, p_on = simulate_onoff_aggregate(
            n_sources, T + WARMUP, ARRIVAL_DT, alpha, alpha,
            xm_on=0.001, xm_off=0.001, rate_on=rate_on,
            device=device, seed=seed, stationary_start=True)
        d = rate.cpu().numpy() * ARRIVAL_DT
        del rate
        if device.type == "cuda":
            torch.cuda.empty_cache()
        out.append(d[int(round(WARMUP / ARRIVAL_DT)):])
    return np.stack(out), p_on


def aggregate(demand, g):
    """Sum every `g` consecutive 1 ms bins -> one plan-period sample."""
    n = demand.shape[-1] // g * g
    return demand[..., :n].reshape(demand.shape[0], -1, g).sum(axis=-1)


# ----------------------------------------------------------------------------
# Forecasters -- one prediction of the next period's demand, per cell
# ----------------------------------------------------------------------------

def fit_arfima_filters(samples, sample_dt, train_frac, T_taps, p):
    """One frozen ARFIMA filter per cell, fitted on that cell's training prefix.

    Returns (weights (n_cells, T_taps), norm) where norm[i] = (m, a) is the
    Norros normalization Z = (x - m*dt) / sqrt(m*a) the filter operates in.
    """
    from farima_f import arfima_weights

    W, norm = [], []
    for i, s in enumerate(samples):
        train = s[:int(len(s) * train_frac)]
        m, a, H = analyze_traffic_model(train, sample_dt)
        Z = (train - m * sample_dt) / np.sqrt(m * a)
        w, d, phi = arfima_weights(Z, T_taps, p=p, horizon=1)
        print(f"  cell{i}: H={H:.3f}  ARFIMA d={d:.4f}  phi={np.round(phi, 3)}"
              f"  sum(w)={w.sum():.3f}")
        W.append(w)
        norm.append((m, a))
    return np.stack(W), norm


def arfima_predict(hist, W, norm, sample_dt, deshrink=False):
    out = np.empty(len(W))
    for i, (m, a) in enumerate(norm):
        scale = np.sqrt(m * a)
        Z = (hist[i][::-1] - m * sample_dt) / scale
        dev = W[i] @ Z
        if deshrink:
            s = W[i].sum()
            if abs(s) > 1e-6:
                dev = dev / s
        out[i] = dev * scale + m * sample_dt
    return out


# ----------------------------------------------------------------------------
# Slot allocation
# ----------------------------------------------------------------------------

def allocate(weights, n_slots, carry=None):
    w = np.clip(np.asarray(weights, dtype=float), 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w)
    exact = w / w.sum() * n_slots
    if carry is None:
        desired = exact
    else:
        # clip guards the case where a large debt would push a share negative;
        # the zero-sum renormalisation at the end absorbs the rounding it costs.
        desired = np.clip(exact + carry, 0.0, None)

    base = np.floor(desired).astype(int)
    left = int(n_slots - base.sum())
    frac = desired - base
    if left > 0:
        base[np.argsort(-frac)[:left]] += 1
    elif left < 0:
        # only cells that actually hold a slot can give one back
        givable = np.where(base > 0)[0]
        order = givable[np.argsort(frac[givable])]
        base[order[:-left]] -= 1

    if carry is not None:
        carry[:] = desired - base
        carry -= carry.mean()          # keep sum(carry) == 0
    return base


def slot_order(counts):
    """Interleave the per-cell slot counts into a beam schedule of len sum(counts).

    Round-robin rather than [0,0,0,1,1,2,...]: with a 15 ms period a cell given
    5 slots is served roughly every 3 ms instead of in one burst at the start,
    which keeps the peak occupancy down without changing anyone's total service.
    """
    n = int(counts.sum())
    plan = np.empty(n, dtype=np.int64)
    left = counts.astype(float).copy()
    credit = np.zeros(len(counts))
    for k in range(n):
        credit += counts                      # each cell accrues its own rate
        credit[left <= 0] = -np.inf           # already fully served
        c = int(np.argmax(credit))
        plan[k] = c
        credit[c] -= n
        left[c] -= 1.0
    return plan


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

def simulate(demand, samples, g, planner, start_period, W=None, norm=None,
             sample_dt=None, T_taps=0, backlog_aware=False, capacity=None,
             keep_trace=True, decim=1):
    
    n_cells, n_periods = samples.shape
    drain = BEAM_RATE * ARRIVAL_DT
    q = np.zeros(n_cells)

    n_ticks = (n_periods - start_period) * g
    occ = (np.empty((n_cells, (n_ticks + decim - 1) // decim))
           if keep_trace else None)
    occ_sum = 0.0
    occ_sq = 0.0
    occ_max = 0.0
    served = np.zeros(n_cells)
    dropped = np.zeros(n_cells)
    offered = 0.0
    waste_avoid = 0.0
    waste_unavoid = 0.0
    fc_log, truth_log, slot_log = [], [], []
    carry = np.zeros(n_cells)          # fractional slots owed, see allocate()

    k = 0
    for t in range(start_period, n_periods):
        hist = samples[:, t - T_taps:t] if T_taps else None
        if planner == "equal":
            wgt = np.ones(n_cells)
        elif planner == "naive":
            wgt = samples[:, t - 1].copy()
        elif planner == "oracle":
            wgt = samples[:, t].copy()
        elif planner == "arfima":
            wgt = arfima_predict(hist, W, norm, sample_dt)
        elif planner == "arfima_ds":
            wgt = arfima_predict(hist, W, norm, sample_dt, deshrink=True)
        else:
            raise ValueError(planner)

        if backlog_aware:
            # Serve what is already queued *plus* what is expected to arrive.
            # Strictly better information, but it is no longer a pure test of
            # the forecast, so it is off by default.
            wgt = np.clip(wgt, 0.0, None) + q

        counts = allocate(wgt, g, carry)
        plan = slot_order(counts)
        fc_log.append(wgt)
        truth_log.append(samples[:, t])
        slot_log.append(counts)

        for j, c in enumerate(plan):
            a = demand[:, t * g + j]           # 1 ms of arrivals, every cell
            offered += a.sum()
            q += a
            if capacity is not None:
                over = np.maximum(q - capacity, 0.0)
                dropped += over
                q -= over

            s = min(q[c], drain)               # the beam serves one cell
            shortfall = (drain - s) / drain
            if shortfall > 0:
                # Was there backlog anywhere else this slot could have gone to?
                other = q.sum() - q[c]
                if other > 0:
                    waste_avoid += shortfall
                else:
                    waste_unavoid += shortfall
            q[c] -= s
            served[c] += s

            if keep_trace and k % decim == 0:
                occ[:, k // decim] = q
            tot = q.sum()
            occ_sum += tot
            occ_sq += tot * tot
            occ_max = max(occ_max, tot)
            k += 1

    return dict(occupancy=occ, served=served, dropped=dropped, offered=offered,
                waste_avoidable=waste_avoid, waste_unavoidable=waste_unavoid,
                n_slots=k, occ_mean=occ_sum / k, occ_max=occ_max,
                occ_sd=np.sqrt(max(occ_sq / k - (occ_sum / k) ** 2, 0.0)),
                forecast=np.array(fc_log), truth=np.array(truth_log),
                slots=np.array(slot_log), q_final=q)


def metrics(name, res):
    """Scalar metrics for one policy. One row of the metrics CSV."""
    n = res["n_slots"]
    occ = res["occupancy"]
    tot = occ.sum(axis=0) if occ is not None else None

    row = dict(policy=name)
    row["loss_ratio"] = res["dropped"].sum() / res["offered"]
    row["dropped_Mb"] = res["dropped"].sum() / 1e6
    row["waste_avoidable_pct"] = 100.0 * res["waste_avoidable"] / n
    row["waste_unavoidable_pct"] = 100.0 * res["waste_unavoidable"] / n
    row["served_Mbps"] = res["served"].sum() / (n * ARRIVAL_DT) / 1e6
    row["backlog_mean_Mb"] = res["occ_mean"] / 1e6
    row["backlog_sd_Mb"] = res["occ_sd"] / 1e6
    row["backlog_max_Mb"] = res["occ_max"] / 1e6
    if tot is not None:
        row["backlog_p95_Mb"] = np.percentile(tot, 95) / 1e6
        row["backlog_p99_Mb"] = np.percentile(tot, 99) / 1e6
    n_cells = len(res["served"])
    for i in range(n_cells):
        row[f"drop_c{i}_Mb"] = res["dropped"][i] / 1e6


    cm, cs, cx = (res.get("cell_mean"), res.get("cell_sd"), res.get("cell_max"))
    if cm is not None:
        for i in range(n_cells):
            row[f"backlog_mean_c{i}_Mb"] = cm[i] / 1e6
            row[f"backlog_sd_c{i}_Mb"] = cs[i] / 1e6
            row[f"backlog_max_c{i}_Mb"] = cx[i] / 1e6
        # Scalar summaries over cells: the peak any single buffer reached (what
        # overflows first), and the worst per-cell time-average.
        row["backlog_max_cell_Mb"] = float(np.max(cx)) / 1e6
        row["backlog_mean_cell_max_Mb"] = float(np.max(cm)) / 1e6
        # Imbalance: 0 when every cell holds the same backlog, growing as the
        # split misallocates. A forecast that is wrong in the right direction
        # can keep the total flat while pushing this up.
        row["backlog_imbalance_Mb"] = float(np.max(cm) - np.min(cm)) / 1e6
    if occ is not None:
        for i in range(n_cells):
            row[f"backlog_p95_c{i}_Mb"] = np.percentile(occ[i], 95) / 1e6
            row[f"backlog_p99_c{i}_Mb"] = np.percentile(occ[i], 99) / 1e6

    f, y = res["forecast"], res["truth"]

    mae = np.abs(f - y).mean(axis=0)                       # per cell
    naive = np.abs(np.diff(y, axis=0)).mean(axis=0)        # per cell
    ss = ((y - y.mean(axis=0)) ** 2).sum(axis=0)      # per cell, own mean


    row["forecast_mase"] = (
        np.nan if name == "equal"
        else float(mae.sum() / max(naive.sum(), 1e-300)))


    ssf = ((f - f.mean(axis=0)) ** 2).sum(axis=0)     # per cell, own mean
    row["forecast_sd_ratio"] = (
        np.nan if name == "equal"
        else float(np.sqrt(ssf.sum() / max(ss.sum(), 1e-300))))
    ideal = (y / y.sum(axis=1, keepdims=True)
             * res["slots"].sum(axis=1, keepdims=True))
    serr = np.abs(res["slots"] - ideal).mean(axis=0)   # per cell
    row["slot_err"] = serr.mean()

    if name != "equal":
        per = mae / np.maximum(naive, 1e-300)
        sdr = np.sqrt(ssf / np.maximum(ss, 1e-300))
        for i in range(len(per)):
            row[f"forecast_mase_c{i}"] = float(per[i])
            row[f"forecast_sd_ratio_c{i}"] = float(sdr[i])
        # Equal-weighted: every cell counts once, whatever its scale.
        row["forecast_mase_eq"] = float(per.mean())
        row["forecast_mase_worst"] = float(per.max())
        row["forecast_sd_ratio_eq"] = float(sdr.mean())
    for i in range(len(serr)):
        row[f"slot_err_c{i}"] = serr[i]


    granted = res["slots"] * BEAM_RATE * ARRIVAL_DT
    row["deficit_Mb_per_period"] = np.clip(y - granted, 0, None).mean() / 1e6
    row["surplus_Mb_per_period"] = np.clip(granted - y, 0, None).mean() / 1e6
    return row


def report(row):
    print(f"\n{row['policy']}")
    print(f"  loss ratio      {row['loss_ratio']:.3e}   "
          f"({row['dropped_Mb']:.3f} Mb dropped)")
    print(f"  wasted slots    {row['waste_avoidable_pct']:6.2f}% avoidable "
          f"(plan error)   {row['waste_unavoidable_pct']:6.2f}% unavoidable "
          f"(system empty)")
    print(f"  served          {row['served_Mbps']:.1f} Mb/s")
    print(f"  backlog         mean {row['backlog_mean_Mb']:8.3f} Mb   "
          f"p95 {row.get('backlog_p95_Mb', float('nan')):8.3f}   "
          f"p99 {row.get('backlog_p99_Mb', float('nan')):8.3f}   "
          f"max {row['backlog_max_Mb']:8.3f}")
    # Count by probing for the indexed keys rather than pattern-matching the
    # names: "backlog_mean_cell_max_Mb" also starts with "backlog_mean_c".
    ncell = 0
    while f"backlog_mean_c{ncell}_Mb" in row:
        ncell += 1
    for i in range(ncell):
        print(f"    cell{i}        mean {row[f'backlog_mean_c{i}_Mb']:8.3f} Mb   "
              f"p95 {row.get(f'backlog_p95_c{i}_Mb', float('nan')):8.3f}   "
              f"p99 {row.get(f'backlog_p99_c{i}_Mb', float('nan')):8.3f}   "
              f"max {row[f'backlog_max_c{i}_Mb']:8.3f}")
    if ncell:
        print(f"    worst cell    max {row['backlog_max_cell_Mb']:8.3f} Mb   "
              f"imbalance {row['backlog_imbalance_Mb']:.3f} Mb")
    print(f"  forecast MASE   {row['forecast_mase']:.4f}   slot err "
          f"{row['slot_err']:.3f} slots/cell/period")
    print(f"  mismatch        deficit {row['deficit_Mb_per_period']:.4f} Mb / "
          f"surplus {row['surplus_Mb_per_period']:.4f} Mb per cell-period")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--T", type=float, default=100.0, help="horizon [s]")
    ap.add_argument("--sample-dt", type=float, default=0.015,
                    help="planning period [s]; must be a multiple of 1 ms")
    ap.add_argument("--n-cells", type=int, default=3)
    ap.add_argument("--n-sources", type=int, default=100,
                    help="ON/OFF sources per cell (same in every cell)")
    ap.add_argument("--alpha", type=float, default=1.04,
                    help="Pareto tail index, both ON and OFF; H=(3-alpha)/2")
    ap.add_argument("--rate-on", type=float, default=1e6, help="b/s per source")
    ap.add_argument("--beam-rate", type=float, default=270e6,
                    help="b/s served while the beam is on a cell")
    ap.add_argument("--past-window", type=int, default=24,
                    help="filter taps / history the planner sees")
    ap.add_argument("--p", type=int, default=2, help="ARFIMA AR order")
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="leading fraction of the trace used to fit ARFIMA; "
                         "the simulation is scored on the remainder")
    ap.add_argument("--planners", nargs="+",
                    default=["equal", "naive", "arfima", "oracle"])
    ap.add_argument("--backlog-aware", action="store_true")
    ap.add_argument("--buffer-mb", type=float, default=BUFFER_MB,
                    help="per-cell buffer capacity [MiB]. 0 = infinite.")
    ap.add_argument("--sweep-mb", type=float, nargs="*",
                    default=[0.25, 0.5, 1, 2, 5, 10, 20, 50],
                    help="buffer sizes for the loss-vs-capacity sweep [MiB]; "
                         "empty list disables the sweep")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[101, 202, 303, 404, 505])
    ap.add_argument("--trace-decim", type=int, default=1,
                    help="store only every d-th 1ms tick of the backlog trace "
                         "(plots/CSV/percentiles only; the mean, sd and max "
                         "are always accumulated over every tick)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="add this to every seed -- the cheapest way to get an\n"
                         "independent realization of the same regime")
    ap.add_argument("--out-prefix", default="beam_hop")
    args = ap.parse_args()

    global BEAM_RATE
    BEAM_RATE = args.beam_rate

    g = int(round(args.sample_dt / ARRIVAL_DT))
    if abs(args.sample_dt / ARRIVAL_DT - g) > 1e-9:
        raise SystemExit("--sample-dt must be a whole number of milliseconds")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  cells={args.n_cells}  X={args.n_sources}/cell  "
          f"alpha={args.alpha} (H={(3-args.alpha)/2:.3f})  T={args.T:g}s")
    print(f"plan period {args.sample_dt*1e3:.0f}ms = {g} slots of 1ms   "
          f"beam {BEAM_RATE/1e6:.0f} Mb/s -> {BEAM_RATE*ARRIVAL_DT/1e3:.0f} kb/slot")

    demand, p_on = generate_cells(args.n_cells, args.T, args.alpha,
                                  args.n_sources, args.rate_on,
                                  [s + args.seed_offset for s in args.seeds],
                                  device)
    samples = aggregate(demand, g)
    n_periods = samples.shape[1]
    demand = demand[:, :n_periods * g]

    offered = demand.sum() / args.T
    print(f"offered {offered/1e6:.1f} Mb/s total "
          f"({demand.sum(axis=1).mean()/args.T/1e6:.1f} per cell, p_on={p_on:.3f})"
          f"   system rho = {offered/BEAM_RATE:.3f}")
    if offered >= BEAM_RATE:
        print("  !! rho >= 1: the single beam cannot carry this load; every "
              "planner will diverge. Lower --n-sources or --rate-on.")

    start = max(args.past_window, int(n_periods * args.train_frac))
    print(f"periods: {n_periods} total, fitting on the first "
          f"{int(n_periods*args.train_frac)}, simulating "
          f"{n_periods-start} ({(n_periods-start)*args.sample_dt:.1f}s)")

    W = norm = None
    if any(k.startswith("arfima") for k in args.planners):
        print("fitting ARFIMA filters (once, frozen):")
        W, norm = fit_arfima_filters(samples, args.sample_dt, args.train_frac,
                                     args.past_window, args.p)

    cap = args.buffer_mb * MIB * 8.0 if args.buffer_mb > 0 else None
    if cap is None:
        print("buffer: infinite (no loss possible)")
    else:
        print(f"buffer: {args.buffer_mb:g} MiB = {cap/1e6:.2f} Mbit per cell "
              f"= {cap/BEAM_RATE*1e3:.0f} ms of full beam rate")

    results, rows = {}, []
    for planner in args.planners:
        results[planner] = simulate(
            demand, samples, g, planner, start, W, norm, args.sample_dt,
            args.past_window, args.backlog_aware, capacity=cap,
            decim=args.trace_decim)
        rows.append(metrics(planner, results[planner]))
        report(rows[-1])

    met = pd.DataFrame(rows).set_index("policy")

    # How much of the achievable gain each planner captured: 0 = no better than
    # the fixed even split, 1 = as good as knowing the future exactly. Only
    # meaningful when equal and oracle actually differ on that metric.
    if "equal" in met.index and "oracle" in met.index:
        print("\nfraction of the oracle gain captured "
              "(0 = no-forecast baseline, 1 = perfect forecast):")
        for key in ["backlog_mean_Mb", "loss_ratio", "waste_avoidable_pct"]:
            lo, hi = met.at["equal", key], met.at["oracle", key]
            if abs(lo - hi) < 1e-15:
                print(f"  {key:<22} equal and oracle tie ({lo:.4g}) -- "
                      f"nothing to capture at this load")
                continue
            gains = "  ".join(
                f"{p} {(lo - met.at[p, key]) / (lo - hi):+.3f}"
                for p in met.index if p not in ("equal", "oracle"))
            print(f"  {key:<22} {gains}")

    met.to_csv(f"{args.out_prefix}_metrics.csv")
    print(f"\nsaved {args.out_prefix}_metrics.csv")

    # Per-period forecast/allocation log: one row per (policy, period, cell).
    per_period = []
    for k, r in results.items():
        n_per, n_c = r["truth"].shape
        for i in range(n_c):
            per_period.append(pd.DataFrame({
                "policy": k, "period": np.arange(n_per), "cell": i,
                "t": np.arange(n_per) * args.sample_dt,
                "demand_bits": r["truth"][:, i],
                "weight": r["forecast"][:, i],
                "slots": r["slots"][:, i]}))
    pd.concat(per_period).to_csv(f"{args.out_prefix}_periods.csv", index=False)
    print(f"saved {args.out_prefix}_periods.csv")

    if args.sweep_mb:
        print("\nloss ratio vs per-cell buffer capacity:")
        hdr = "  ".join(f"{b:>9g}" for b in args.sweep_mb)
        print(f"  {'MiB':<10} {hdr}")
        sweep = []
        for planner in args.planners:
            vals = []
            for b in args.sweep_mb:
                r = simulate(demand, samples, g, planner, start, W, norm,
                             args.sample_dt, args.past_window,
                             args.backlog_aware,
                             capacity=b * MIB * 8.0, keep_trace=False)
                loss = r["dropped"].sum() / r["offered"]
                vals.append(loss)
                sweep.append(dict(policy=planner, buffer_MiB=b,
                                  buffer_Mbit=b * MIB * 8.0 / 1e6,
                                  buffer_ms_of_beam=b * MIB * 8.0 / BEAM_RATE * 1e3,
                                  loss_ratio=loss,
                                  waste_avoidable_pct=100.0 * r["waste_avoidable"]
                                  / r["n_slots"]))
            print(f"  {planner:<10} " + "  ".join(f"{v:>9.2e}" for v in vals))
        sweep = pd.DataFrame(sweep)
        sweep.to_csv(f"{args.out_prefix}_loss_vs_buffer.csv", index=False)
        print(f"saved {args.out_prefix}_loss_vs_buffer.csv")
    else:
        sweep = None

    # traces
    t = (np.arange(results[args.planners[0]]["occupancy"].shape[1])
         * ARRIVAL_DT * args.trace_decim)
    cols = {"t": t}
    for k, r in results.items():
        for i in range(args.n_cells):
            cols[f"{k}_c{i}"] = r["occupancy"][i]
    pd.DataFrame(cols).to_csv(f"{args.out_prefix}.csv", index=False)
    print(f"\nsaved {args.out_prefix}.csv")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Zoom on 200 plan periods, or the whole run if it is shorter.
    zoom_s = min(t[-1], 200 * args.sample_dt)
    n_zoom = max(2, int(round(zoom_s / (ARRIVAL_DT * args.trace_decim))))
    subtitle = (f"{args.n_cells} cells / 1 beam, "
                f"{args.sample_dt*1e3:g}ms plans, alpha={args.alpha}, "
                f"rho={offered/BEAM_RATE:.3f}, buffer {args.buffer_mb:g} MiB")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    for k, r in results.items():
        axes[0].plot(t, r["occupancy"].sum(axis=0) / 1e6, lw=0.4, label=k)
        axes[1].plot(t[:n_zoom], r["occupancy"].sum(axis=0)[:n_zoom] / 1e6,
                     lw=0.8, label=k)
    axes[0].set_ylabel("total backlog [Mb]")
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("total backlog [Mb]")
    axes[1].set_xlabel(f"time [s]  (first {zoom_s:g}s)")
    fig.suptitle("Beam-hopping backlog -- " + subtitle)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=120)
    print(f"saved {args.out_prefix}.png")

    # One figure per policy: aggregate backlog on top, the three cells that
    # make it up below. The combined plot above hides which cell is drowning,
    # and with one beam that is usually the whole story.
    ymax = max(r["occupancy"].sum(axis=0).max() for r in results.values()) / 1e6
    for k, r in results.items():
        occ = r["occupancy"] / 1e6
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, occ.sum(axis=0), lw=0.4, color="k")
        axes[0].set_ylabel("total backlog [Mb]")
        axes[0].set_ylim(0, ymax * 1.05)      # shared scale, so policies compare
        if cap is not None:
            axes[0].axhline(args.n_cells * cap / 1e6, color="r", lw=0.8,
                            ls="--", label="aggregate capacity")
            axes[0].legend(fontsize=8)
        for i in range(args.n_cells):
            axes[1].plot(t, occ[i], lw=0.4, label=f"cell{i}")
        if cap is not None:
            axes[1].axhline(cap / 1e6, color="r", lw=0.8, ls="--",
                            label="per-cell capacity")
        axes[1].set_ylabel("per-cell backlog [Mb]")
        axes[1].set_xlabel("time [s]")
        axes[1].legend(fontsize=8, ncol=4)
        row = met.loc[k]
        fig.suptitle(f"backlog: {k} -- " + subtitle + "\n"
                     f"mean {row['backlog_mean_Mb']:.3f} Mb   "
                     f"max {row['backlog_max_Mb']:.3f} Mb   "
                     f"loss {row['loss_ratio']:.3e}", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_backlog_{k}.png", dpi=120)
        plt.close(fig)
        print(f"saved {args.out_prefix}_backlog_{k}.png")

    if sweep is not None:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
        for k in args.planners:
            s = sweep[sweep.policy == k]
            ax[0].loglog(s.buffer_MiB, np.maximum(s.loss_ratio, 1e-9),
                         marker="o", ms=3, label=k)
        if cap is not None:
            ax[0].axvline(args.buffer_mb, color="k", lw=0.6, ls=":")
        ax[0].set_xlabel("per-cell buffer [MiB]")
        ax[0].set_ylabel("loss ratio (floored at 1e-9)")
        ax[0].set_title("Loss vs buffer capacity")
        ax[0].legend(fontsize=8)
        ax[0].grid(alpha=0.3)

        bars = met.loc[args.planners]
        x = np.arange(len(bars))
        ax[1].bar(x - 0.2, bars.waste_avoidable_pct, 0.4, label="avoidable")
        ax[1].bar(x + 0.2, bars.waste_unavoidable_pct, 0.4, label="unavoidable")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(bars.index, fontsize=8)
        ax[1].set_ylabel("% of slots wasted")
        ax[1].set_title(f"Wasted beam time (buffer {args.buffer_mb:g} MiB)")
        ax[1].legend(fontsize=8)
        ax[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(f"{args.out_prefix}_loss.png", dpi=120)
        print(f"saved {args.out_prefix}_loss.png")


if __name__ == "__main__":
    main()
