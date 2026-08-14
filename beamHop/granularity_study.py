#   python granularity_study.py                       # the full study
#   python granularity_study.py --test-hours 0.25 --smoke   # quick check
#   python granularity_study.py --no-backlog-aware --sweep-mb 1 10 100

import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch

import beam_hopping_sim as B
from beam_hopping_sim import (ARRIVAL_DT, MIB, allocate, slot_order,
                              analyze_traffic_model, simulate_onoff_aggregate)
from gen_onoff_pareto import local_whittle_H
from norros_f import select_H


H_GRID = np.arange(0.50, 0.9951, 0.005)
SEED_BASE = 101
TRAIN_OFFSET = 500000

INIT_OFFSET = 900000


def seeds_for(rep, n_cells, base=SEED_BASE):
    """(test_seeds, train_seeds) for replicate `rep` -- n_cells each."""
    lo = base + rep * n_cells
    return ([lo + i for i in range(n_cells)],
            [lo + TRAIN_OFFSET + i for i in range(n_cells)])


# ----------------------------------------------------------------------------
# Traffic
# ----------------------------------------------------------------------------

def generate(seeds, n_cells, seconds, dt, alphas, n_sources, rate_on, device):
    out = []
    for seed, alpha, X in zip(seeds[:n_cells], alphas, n_sources):
        rate, _ = simulate_onoff_aggregate(
            X, seconds, dt, alpha, alpha, xm_on=0.001, xm_off=0.001,
            rate_on=rate_on, device=device, seed=seed, stationary_start=True)
        out.append(rate.cpu().numpy() * dt)
        del rate
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return np.stack(out)


# ----------------------------------------------------------------------------
# Simulation -- same dynamics as beam_hopping_sim.simulate, but the hot loop
# works on python floats instead of 3-element numpy arrays. At 3.6e7 ticks per
# policy the numpy overhead dominated; this is ~4x faster and bit-identical.
# ----------------------------------------------------------------------------

def run(demand, samples, g, planner, start, capacity, W=None, norm=None,
        sample_dt=None, T_taps=0, adaptive_mean=True, decim=1,
        extra_caps=(), backlog_aware=True, fixed_weights=None, nets=None):
    
    if backlog_aware and extra_caps:
        raise ValueError("backlog_aware invalidates the shared-pass buffer "
                         "sweep: the plan depends on the queue, so each "
                         "capacity needs its own pass")
    n_cells, n_periods = samples.shape
    drain = B.BEAM_RATE * ARRIVAL_DT
    q = [0.0] * n_cells
    dropped = [0.0] * n_cells
    served = [0.0] * n_cells
    offered = 0.0
    waste_av = 0.0
    waste_un = 0.0
    cap = float("inf") if capacity is None else capacity

    # Parallel buffer systems: queues, drops and avoidable waste per extra cap.
    xcaps = [float(c) for c in extra_caps]
    xq = [[0.0] * n_cells for _ in xcaps]
    xdrop = [0.0] * len(xcaps)
    xwaste = [0.0] * len(xcaps)

    n_ticks = (n_periods - start) * g
    occ = np.empty((n_cells, (n_ticks + decim - 1) // decim))
    occ_sum = occ_sq = occ_max = 0.0

    q_sum = [0.0] * n_cells
    q_sq = [0.0] * n_cells
    q_max = [0.0] * n_cells

    # Running mean of the observed demand, seeded with the history the planner
    # already has at t=start. Only the mean adapts; d, phi and a stay frozen.
    run_sum = samples[:, max(0, start - T_taps):start].sum(axis=1).astype(float)
    run_n = float(min(T_taps, start))

    fx = (np.ones(n_cells) if fixed_weights is None
          else np.asarray(fixed_weights, dtype=float))
    if fx.shape != (n_cells,):
        raise ValueError(f"fixed_weights must have {n_cells} entries, got {fx}")
    if fx.min() < 0 or fx.sum() <= 0:
        raise ValueError(f"fixed_weights must be non-negative and non-zero, "
                         f"got {fx}")

    pre_fc = None
    if planner == "dlinear":
        if nets is None:
            raise ValueError("planner 'dlinear' needs the trained nets")
        from dlinear_f import predict_batch
        pre_fc = np.empty((n_periods - start, n_cells))
        ts = np.arange(start, n_periods)

        idx = np.arange(T_taps)[None, :] + (ts - T_taps)[:, None]
        for i in range(n_cells):
            net, mu, sd = nets[i]
            Xw = (samples[i][idx].astype(float) - mu) / sd
            pre_fc[:, i] = predict_batch(net, Xw) * sd + mu

    fc_log, truth_log, slot_log = [], [], []
    carry = np.zeros(n_cells)          # fractional slots owed, see allocate()
    dem = demand                      # local refs: attribute lookup is not free
    k = 0
    for t in range(start, n_periods):
        truth = samples[:, t]
        if planner == "fixed":
            wgt = fx
        elif planner == "naive":
            wgt = samples[:, t - 1].astype(float)
        elif planner == "oracle":
            wgt = truth.astype(float)
        elif planner in ("arfima", "linearp"):

            hist = samples[:, t - T_taps:t]
            wgt = np.empty(n_cells)
            for i in range(n_cells):
                m_tr, a = norm[i]
                m = (run_sum[i] / run_n / sample_dt) if (adaptive_mean and
                                                         run_n > 0) else m_tr
                scale = np.sqrt(m_tr * a)      # a is a shape, keep it frozen
                Z = (hist[i][::-1] - m * sample_dt) / scale
                wgt[i] = (W[i] @ Z) * scale + m * sample_dt
        elif planner == "dlinear":
            # Precomputed above -- identical arithmetic to the branch overhead,
            # just hoisted out of the loop. See the pre_fc block.
            wgt = pre_fc[t - start]
        else:
            raise ValueError(planner)

        awgt = (np.clip(wgt, 0.0, None) + q
                if backlog_aware and planner != "fixed" else wgt)

        counts = allocate(awgt, g, carry)
        plan = slot_order(counts)
        fc_log.append(wgt)
        truth_log.append(truth)
        slot_log.append(counts)

        run_sum += truth
        run_n += 1.0

        base = t * g
        for j in range(g):
            c = plan[j]
            tot = 0.0
            for i in range(n_cells):
                a_i = dem[i, base + j]
                offered += a_i
                v = q[i] + a_i
                if v > cap:
                    dropped[i] += v - cap
                    v = cap
                q[i] = v
                tot += v
                # The served cell is excluded here and accumulated below,
                # after its drain, so every cell contributes its post-service
                # occupancy exactly once -- matching how `tot` is accumulated.
                if i != c:
                    q_sum[i] += v
                    q_sq[i] += v * v
                    if v > q_max[i]:
                        q_max[i] = v
            qc = q[c]
            s = qc if qc < drain else drain
            if s < drain:
                if tot - qc > 0.0:
                    waste_av += (drain - s) / drain
                else:
                    waste_un += (drain - s) / drain
            vc = qc - s
            q[c] = vc
            served[c] += s
            tot -= s
            q_sum[c] += vc
            q_sq[c] += vc * vc
            if vc > q_max[c]:
                q_max[c] = vc

            for x in range(len(xcaps)):
                xc, qx = xcaps[x], xq[x]
                xtot = 0.0
                for i in range(n_cells):
                    v = qx[i] + dem[i, base + j]
                    if v > xc:
                        xdrop[x] += v - xc
                        v = xc
                    qx[i] = v
                    xtot += v
                qcx = qx[c]
                sx = qcx if qcx < drain else drain
                if sx < drain and xtot - qcx > 0.0:
                    xwaste[x] += (drain - sx) / drain
                qx[c] = qcx - sx
            if k % decim == 0:
                occ[:, k // decim] = q
            occ_sum += tot
            occ_sq += tot * tot
            if tot > occ_max:
                occ_max = tot
            k += 1

    return dict(occupancy=occ, served=np.array(served),
                dropped=np.array(dropped), offered=offered,
                waste_avoidable=waste_av, waste_unavoidable=waste_un,
                n_slots=k, occ_mean=occ_sum / k, occ_max=occ_max,
                occ_sd=np.sqrt(max(occ_sq / k - (occ_sum / k) ** 2, 0.0)),
                cell_mean=np.array(q_sum) / k, cell_max=np.array(q_max),
                cell_sd=np.sqrt(np.maximum(
                    np.array(q_sq) / k - (np.array(q_sum) / k) ** 2, 0.0)),
                forecast=np.array(fc_log), truth=np.array(truth_log),
                slots=np.array(slot_log), q_final=np.array(q),
                extra={c: dict(loss_ratio=xdrop[x] / offered,
                               waste_avoidable_pct=100.0 * xwaste[x] / k)
                       for x, c in enumerate(xcaps)})


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-dts", type=float, nargs="+",
                    default=[0.015, 1, 60.0])
    ap.add_argument("--test-hours", type=float, default=10.0,
                    help="test window [h]; 10 h = 3.6e7 bins = 0.81 GB on the "
                         "GPU (measured). 45 h will OOM a 4 GB card.")
    ap.add_argument("--train-samples", type=int, default=4200)
    ap.add_argument("--n-cells", type=int, default=3)
    ap.add_argument("--n-sources", type=int, nargs="+", default=[180],
                    help="users per cell: one value (all cells) or one per "
                         "cell. Keep the SUM fixed across scenarios to vary "
                         "load asymmetry at constant rho.")
    ap.add_argument("--alpha", type=float, nargs="+", default=[1.04],
                    help="Pareto tail index: one value (all cells) or one per "
                         "cell, e.g. --alpha 1.04 1.24 1.44 for a mixed "
                         "scenario. H = (3-alpha)/2. Does not affect offered "
                         "load, so rho is the same for every alpha.")
    ap.add_argument("--rate-on", type=float, default=1e6)
    ap.add_argument("--beam-rate", type=float, default=270e6)
    ap.add_argument("--past-window", type=int, default=24)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--buffer-mb", type=float, default=2.0)
    ap.add_argument("--sweep-mb", type=float, nargs="*", default=[],
                    help="extra buffer sizes [MiB] evaluated in the same pass; "
                         "empty means only --buffer-mb is simulated")
    ap.add_argument("--planners", nargs="+",
                    default=["fixed", "naive", "linearp"],
                    choices=["fixed", "naive", "linearp", "dlinear", "arfima",
                             "oracle"],
                    help="'arfima' and 'oracle' are still implemented and can "
                         "be requested explicitly; they are out of the default "
                         "set because the study reports fixed/naive/linearP.")
    ap.add_argument("--fixed-weights", type=float, nargs="+", default=None,
                    help="static split for the 'fixed' planner: one weight per "
                         "cell, normalized internally. Default is the even "
                         "split. Under asymmetric user counts pass weights "
                         "proportional to the users (e.g. 6 3 1 for "
                         "900/450/150) so the reference is the best static "
                         "allocation rather than a deliberately bad one.")
    # DLinear training. Defaults are deliberately modest: on one channel the
    # model is 2*past_window+2 parameters, so it converges in a few dozen
    # epochs and early stopping usually fires well before the cap.
    ap.add_argument("--dl-epochs", type=int, default=60)
    ap.add_argument("--dl-lr", type=float, default=1e-3)
    ap.add_argument("--dl-batch", type=int, default=64)
    ap.add_argument("--no-adaptive-mean", action="store_true")
    ap.add_argument("--no-backlog-aware", action="store_true",
                    help="revert to the open-loop split. By default each cell's "
                         "current buffer occupancy is added to its forecast "
                         "(unweighted -- both are bits per period) so the beam "
                         "is sized by what the cell must clear this period. "
                         "Turning it off gives the pure test of the forecast, "
                         "and is required by --sweep-mb, which needs a "
                         "queue-independent plan to share one pass.")
    ap.add_argument("--trace-decim", type=int, default=100)
    ap.add_argument("--out-dir", default="results_new")
    ap.add_argument("--n-replicates", type=int, default=1,
                    help="independent experiments; each draws its own n_cells "
                         "test seeds AND its own n_cells train seeds, so the "
                         "filter is refit per replicate and its estimation "
                         "variance lands inside the pooled error bar")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run: 200 train samples, no buffer sweep")
    args = ap.parse_args()

    # One alpha per cell. A single value broadcasts, so the scalar form keeps
    # working; anything else must name every cell explicitly rather than be
    # silently recycled, since a partial list is far more likely a typo than an
    # intent. `alpha_tag` is what identifies the scenario on disk and in the
    # CSV: "1p04" when uniform, "1p04-1p24-1p44" when mixed.
    if len(args.alpha) == 1:
        args.alpha = args.alpha * args.n_cells
    if len(args.alpha) != args.n_cells:
        ap.error(f"--alpha needs 1 or {args.n_cells} values, "
                 f"got {len(args.alpha)}")
    uniform = len(set(args.alpha)) == 1
    args.alpha_str = ("%g" % args.alpha[0] if uniform
                      else "|".join("%g" % a for a in args.alpha))
    args.alpha_tag = ("%g" % args.alpha[0] if uniform else
                      "-".join("%g" % a for a in args.alpha)).replace(".", "p")

    # Same broadcast-or-name-every-cell rule for the user counts.
    if len(args.n_sources) == 1:
        args.n_sources = args.n_sources * args.n_cells
    if len(args.n_sources) != args.n_cells:
        ap.error(f"--n-sources needs 1 or {args.n_cells} values, "
                 f"got {len(args.n_sources)}")
    if any(x <= 0 for x in args.n_sources):
        ap.error(f"--n-sources must be positive, got {args.n_sources}")
    # The fixed split. Default (None) means the even split; anything else must
    # name every cell, same rule as --alpha and --n-sources.
    if args.fixed_weights is not None:
        if len(args.fixed_weights) == 1:
            args.fixed_weights = args.fixed_weights * args.n_cells
        if len(args.fixed_weights) != args.n_cells:
            ap.error(f"--fixed-weights needs 1 or {args.n_cells} values, "
                     f"got {len(args.fixed_weights)}")
        if min(args.fixed_weights) < 0 or sum(args.fixed_weights) <= 0:
            ap.error(f"--fixed-weights must be non-negative and not all zero, "
                     f"got {args.fixed_weights}")
    args.fw_str = ("even" if args.fixed_weights is None
                   else "|".join("%g" % w for w in args.fixed_weights))

    x_uniform = len(set(args.n_sources)) == 1
    args.x_str = ("%d" % args.n_sources[0] if x_uniform
                  else "|".join("%d" % x for x in args.n_sources))

    B.BEAM_RATE = args.beam_rate
    if args.smoke:
        args.train_samples = 200
        args.sweep_mb = []
    backlog_aware = not args.no_backlog_aware
    if backlog_aware and args.sweep_mb:
        ap.error("--sweep-mb needs a queue-independent plan so the extra "
                 "buffer sizes can share one pass, but the backlog-aware "
                 "split makes the plan depend on the queue. Add "
                 "--no-backlog-aware, or run the sizes as separate passes.")
    adaptive = not args.no_adaptive_mean
    tag = "" if adaptive else "_frozenmean"
    if not backlog_aware:
        tag += "_openloop"
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_s = args.test_hours * 3600.0
    cap = args.buffer_mb * MIB * 8.0 if args.buffer_mb > 0 else None

    warm_s = args.past_window * max(args.sample_dts)
    gen_s = test_s + warm_s
    warm_bins = int(round(warm_s / ARRIVAL_DT))

    print(f"device={device}  cells={args.n_cells}  X={args.x_str}/cell  "
          f"alpha={args.alpha_str} "
          f"(H={'|'.join('%.3f' % ((3-a)/2) for a in args.alpha)})")
    print(f"TEST  {args.n_replicates} replicate(s) x {args.n_cells} cells, "
          f"{test_s:g}s = {args.test_hours:g}h measured @1ms "
          f"(+{warm_s:g}s warm-up) = {gen_s/ARRIVAL_DT:.3g} bins/cell")
    print(f"TRAIN {args.train_samples} samples per granularity, refit per "
          f"replicate with its own seeds, generated at bin width = sample_dt")
    print(f"buffer {args.buffer_mb:g} MiB = "
          f"{cap/1e6:.2f} Mbit/cell   beam {B.BEAM_RATE/1e6:.0f} Mb/s   "
          f"adaptive_mean={adaptive}   backlog_aware={backlog_aware}\n")

    t0 = time.time()
    rows, sweep_rows = [], []

    for rep in range(args.n_replicates):
        test_seeds, train_seeds = seeds_for(rep, args.n_cells, args.seed_base)
        print("#" * 78)
        print(f"REPLICATE {rep}   test seeds {test_seeds}   "
              f"train seeds {train_seeds}")
        t_rep = time.time()
        test = generate(test_seeds, args.n_cells, gen_s, ARRIVAL_DT,
                        args.alpha, args.n_sources, args.rate_on, device)
        # rho is reported over the MEASURED window only -- the warm-up bins are
        # never simulated, so including them would describe a different trace
        # than the one the metrics come from.
        offered = test[:, warm_bins:].sum() / test_s
        rho = offered / B.BEAM_RATE
        print(f"test traffic generated in {time.time()-t_rep:.1f}s   "
              f"offered {offered/1e6:.1f} Mb/s   rho = {rho:.3f}\n")

        for sample_dt in args.sample_dts:
            g = int(round(sample_dt / ARRIVAL_DT))
            samples = B.aggregate(test, g)
            n_per = samples.shape[1]
            start = int(round(warm_s / sample_dt))
            train_s = args.train_samples * sample_dt
            print("=" * 78)
            print(f"[rep {rep}] sample_dt = {sample_dt:g}s   {g} slots/period  "
                  f" {n_per} periods, start={start} "
                  f"({n_per-start} decisions = "
                  f"{(n_per-start)*sample_dt:g}s measured)")
            print(f"  train: {args.train_samples} samples = {train_s:g}s "
                  f"({train_s/3600:.2f}h), generated at {sample_dt:g}s bins")

            H_meas = [local_whittle_H(samples[i, start:])
                      for i in range(args.n_cells)]
            print("  H_meas (local Whittle, test window): "
                  + "  ".join(f"c{i}={h:.3f}" for i, h in enumerate(H_meas))
                  + f"   [nominal {'|'.join('%.2f' % ((3-a)/2) for a in args.alpha)}]")

            filt = [k for k in args.planners
                    if k in ("arfima", "linearp", "dlinear")]
            WS, norm, nets = {}, None, None
            if filt:
                t1 = time.time()
                tr = generate(train_seeds, args.n_cells, train_s, sample_dt,
                              args.alpha, args.n_sources, args.rate_on, device)
                norm = []
                Zs = []
                for i, s in enumerate(tr):
                    m, a, H = analyze_traffic_model(s, sample_dt)
                    Z = (s - m * sample_dt) / np.sqrt(m * a)
                    m_te = samples[i, start:].mean() / sample_dt
                    print(f"    cell{i}: H_est={H:.3f}   "
                          f"m_train={m/1e6:.1f} vs m_test={m_te/1e6:.1f} Mb/s "
                          f"({100*(m-m_te)/m_te:+.1f}%)")
                    norm.append((m, a))
                    Zs.append(Z)

                if "arfima" in filt:
                    from farima_f import arfima_weights
                    W = []
                    for i, Z in enumerate(Zs):
                        w, d, phi = arfima_weights(Z, args.past_window,
                                                   p=args.p, horizon=1)
                        print(f"      arfima  cell{i}: d={d:.4f} "
                              f"sum(w)={w.sum():.3f}")
                        W.append(w)
                    WS["arfima"] = np.stack(W)

                if "linearp" in filt:
                    from linearP_f import linearp_weights
                    W = []
                    for i, Z in enumerate(Zs):
                        H_fit, _, _ = select_H(Z, args.past_window, horizon=1,
                                               grid=H_GRID)
                        w = linearp_weights(args.past_window, H_fit, horizon=1)
                        edge = "" if 0.5 < H_fit < 0.995 else "  <-- GRID EDGE"
                        print(f"      linearP cell{i}: H_fit={H_fit:.3f} "
                              f"sum(w)={w.sum():.3f}{edge}")
                        W.append(w)
                    WS["linearp"] = np.stack(W)

                if "dlinear" in filt:
                    from dlinear_f import train_dlinear
                    nets = []
                    for i, s in enumerate(tr):
                        net, mu, sd, vnmse = train_dlinear(
                            np.asarray(s, dtype=float), args.past_window,
                            horizon=1, epochs=args.dl_epochs, lr=args.dl_lr,
                            batch_size=args.dl_batch, device=device,
                            seed=train_seeds[i] + INIT_OFFSET)
                        print(f"      dlinear cell{i}: val NMSE={vnmse:.4f}  "
                              f"scaler mu={mu/1e6:.2f} Mb sd={sd/1e6:.2f} Mb")
                        nets.append((net, mu, sd))

                print(f"    (train gen + fit: {time.time()-t1:.1f}s)")

            extra = [b * MIB * 8.0 for b in args.sweep_mb]
            results = {}
            for planner in args.planners:
                t1 = time.time()
                r = run(test, samples, g, planner, start, cap,
                        WS.get(planner), norm,
                        sample_dt, args.past_window, adaptive,
                        args.trace_decim, extra_caps=extra,
                        backlog_aware=backlog_aware,
                        fixed_weights=args.fixed_weights, nets=nets)
                for b in args.sweep_mb:
                    e = r["extra"][b * MIB * 8.0]
                    sweep_rows.append(dict(replicate=rep, sample_dt=sample_dt,
                                           policy=planner, buffer_MiB=b, **e))
                results[planner] = r
                row = B.metrics(planner, r)
                row["replicate"] = rep
                row["sample_dt"] = sample_dt
                row["n_decisions"] = n_per - start
                row["train_seconds"] = train_s
                # Constant within a replicate, but carried per row so a CSV
                # read on its own is self-describing.
                row["rho"] = rho
                row["offered_Mbps"] = offered / 1e6
                row["beam_Mbps"] = B.BEAM_RATE / 1e6
                row["buffer_MiB"] = args.buffer_mb
                row["test_hours"] = args.test_hours
                row["alpha"] = args.alpha_str
                for i, a in enumerate(args.alpha):
                    row[f"alpha_c{i}"] = a
                    row[f"H_nominal_c{i}"] = (3.0 - a) / 2.0
                for i, h in enumerate(H_meas):
                    row[f"H_meas_c{i}"] = h
                row["n_sources"] = sum(args.n_sources)
                for i, x in enumerate(args.n_sources):
                    row[f"n_sources_c{i}"] = x
                row["fixed_weights"] = args.fw_str
                row["adaptive_mean"] = adaptive
                row["backlog_aware"] = backlog_aware
                rows.append(row)
                B.report(row)
                print(f"  ({time.time()-t1:.0f}s)")

            if args.sweep_mb:
                sw = pd.DataFrame([s for s in sweep_rows
                                   if s["sample_dt"] == sample_dt
                                   and s["replicate"] == rep])
                print("\n  loss vs buffer [MiB]:")
                piv = sw.pivot(index="policy", columns="buffer_MiB",
                               values="loss_ratio")
                print(piv.to_string(float_format=lambda v: f"{v:.2e}"))

            # Only replicate 0 is plotted -- 10 replicates x 3 granularities x
            # 4 planners would be 120 near-identical PNGs, and the figure is a
            # qualitative illustration of one trace, not a pooled result.
            if rep == 0:
                plot_backlogs(results, sample_dt, args, cap, rows)
                write_summary(rows, sample_dt, args, tag, rep=0)

            # Written as we go, not once at the end: a run that dies on a later
            # replicate still leaves the ones it finished on disk.
            save_tables(rows, sweep_rows, args, tag)
            print()

        del test
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"replicate {rep} done in {time.time()-t_rep:.0f}s\n")

    met = pd.DataFrame(rows)
    print("\nper-replicate summary -- loss ratio and mean backlog:")
    for sample_dt in sorted(met.sample_dt.unique()):
        sub = met[met.sample_dt == sample_dt]
        agg = sub.groupby("policy")[["loss_ratio", "backlog_mean_Mb"]].mean()
        print(f"  sample_dt={sample_dt:g}s  " + "   ".join(
            f"{p}: {agg.at[p,'loss_ratio']:.2e} / "
            f"{agg.at[p,'backlog_mean_Mb']:.2f}Mb" for p in agg.index))

    if args.n_replicates > 1:
        pool(met, args, tag)
    print(f"\ntotal {time.time()-t0:.0f}s")


# ----------------------------------------------------------------------------
# Pooling across replicates
# ----------------------------------------------------------------------------

POOL_COLS = ["loss_ratio", "backlog_mean_Mb", "backlog_sd_Mb",
             "backlog_p95_Mb", "backlog_p99_Mb", "backlog_max_Mb",
             "backlog_max_cell_Mb", "backlog_mean_cell_max_Mb",
             "backlog_imbalance_Mb",
             "waste_avoidable_pct", "served_Mbps", "forecast_mase",
             "forecast_mase_eq", "forecast_mase_worst",
             "forecast_sd_ratio", "forecast_sd_ratio_eq",
             "slot_err", "deficit_Mb_per_period", "surplus_Mb_per_period"]

# Per-cell columns are appended by _pool_cols() rather than listed, since the
# cell count is a run-time argument.
def _pool_cols(df):
    # Indexed cells only: the pattern must not also catch the over-cell
    # summaries (backlog_mean_cell_max_Mb), which are already in POOL_COLS.
    pat = re.compile(r"^(backlog_(mean|sd|max|p95|p99)_c\d+_Mb"
                     r"|forecast_mase_c\d+|forecast_sd_ratio_c\d+"
                     r"|slot_err_c\d+)$")
    extra = sorted(c for c in df.columns if pat.match(c))
    return [c for c in POOL_COLS if c in df.columns] + extra


def _t95(df):
    """Two-sided 95% t quantile. Table to df=30, normal beyond -- scipy is not
    imported here and the study never pools fewer than a handful of seeds."""
    tab = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
           19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
           25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}
    return tab.get(df, 1.96) if df >= 1 else float("nan")


def pool(met, args, tag):
    """Mean +/- standard error across replicates, plus PAIRED comparisons.

    Replicates are independent realizations, so the SE of the pooled mean falls
    as 1/sqrt(R) -- the ordinary rate. That is the whole reason for replicating:
    WITHIN one trace at H ~ 0.98 the mean converges like n^(2H-2) = n^-0.04,
    i.e. not at all, so a longer single run buys almost no precision while
    independent seeds buy the usual amount.

    The planner comparisons are done PAIRED, as per-replicate ratios, because
    every planner in a replicate sees the identical arrivals. The common
    realization effect -- which is large, since it sets how unbalanced the cell
    means happen to be -- cancels in the ratio, so the paired SE is far tighter
    than differencing two independent means would give. Ratios are averaged
    geometrically: they are multiplicative and span orders of magnitude.
    """
    R = met.replicate.nunique()
    cols = _pool_cols(met)
    g = met.groupby(["sample_dt", "policy"])[cols]
    stats = g.agg(["mean", "std", "count"])
    stats.to_csv(f"{args.out_dir}/pooled{tag}.csv")

    lines = [f"pooled over R = {R} replicates "
             f"({args.n_cells} test + {args.n_cells} train seeds each)",
             f"alpha = {args.alpha_str}   "
             f"H = {'|'.join('%.3f' % ((3-a)/2) for a in args.alpha)}",
             f"test {args.test_hours:g}h/replicate   "
             f"{args.x_str} sources/cell   "
             f"fixed split {args.fw_str}   "
             f"beam {args.beam_rate/1e6:.0f} Mb/s   "
             f"buffer {args.buffer_mb:g} MiB",
             "mean +/- SE of the mean; SE = sd/sqrt(R)", ""]

    for dt in sorted(met.sample_dt.unique()):
        sub = met[met.sample_dt == dt]
        lines.append("=" * 72)
        lines.append(f"sample_dt = {dt:g}s   "
                     f"decisions/replicate = {int(sub.n_decisions.iloc[0])}")
        pols = list(dict.fromkeys(sub.policy))
        w = max(18, max(len(p) for p in pols) + 2)
        lines.append("  " + "metric".ljust(24)
                     + "".join(p.rjust(w) for p in pols))
        for c in cols:
            cells = []
            for p in pols:
                v = sub[sub.policy == p][c]
                mu, sd = v.mean(), v.std(ddof=1)
                cells.append(("nan" if mu != mu else
                              f"{mu:.3e}+/-{sd/np.sqrt(len(v)):.0e}").rjust(w))
            lines.append("  " + c.ljust(24) + "".join(cells))

        pols_p = [p for p in pols if p in ("fixed", "naive", "linearp",
                                           "dlinear", "arfima", "oracle")]
        pairs = [(a, b) for i, a in enumerate(pols_p) for b in pols_p[i+1:]]
        if pairs:
            lines.append("")
            lines.append("  PAIRED (same seeds; negative = first policy "
                         "better).  t-interval at 95%")

        for metric, unit in (("backlog_max_cell_Mb", "Mb"),
                             ("backlog_imbalance_Mb", "Mb"),
                             ("backlog_max_Mb", "Mb"),
                             ("backlog_mean_Mb", "Mb"),
                             ("forecast_mase", ""),
                             ("forecast_mase_eq", ""),
                             ("forecast_mase_worst", ""),
                             ("forecast_sd_ratio", ""),
                             ("forecast_sd_ratio_eq", "")):
            if metric not in sub.columns or sub[metric].isna().all():
                continue
            piv = sub.pivot(index="replicate", columns="policy", values=metric)
            for a, b in pairs:
                if a not in piv or b not in piv:
                    continue
                d = (piv[a] - piv[b]).dropna().values
                if len(d) < 2:
                    continue
                mu = d.mean()
                se = d.std(ddof=1) / np.sqrt(len(d))
                half = _t95(len(d) - 1) * se
                sig = "*" if abs(mu) > half else " "
                wins = int((d < 0).sum())
                label = (metric.replace("backlog_", "").replace("_Mb", "")
                         .replace("forecast_", ""))
                lines.append(f"  {sig} d{label}"
                             f"[{a}-{b}] = {mu:+.4f} +/- {half:.4f} {unit}"
                             f"   {a} better in {wins}/{len(d)}")

        piv = sub.pivot(index="replicate", columns="policy",
                        values="loss_ratio")
        for a, b in pairs:
            if a not in piv or b not in piv:
                continue
            r = (piv[a] / piv[b]).values
            r = r[np.isfinite(r) & (r > 0)]
            if not len(r):
                continue
            lg = np.log(r)
            gm = np.exp(lg.mean())
            se = np.exp(lg.mean() + lg.std(ddof=1) / np.sqrt(len(lg))) - gm
            wins = int((r < 1).sum())
            lines.append(f"    loss[{a}]/loss[{b}] = {gm:.3f} "
                         f"(x/ {1+se/gm:.3f})   {a} better in {wins}/{len(r)}")
        lines.append("")

    path = f"{args.out_dir}/summary_pooled{tag}.txt"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"  saved {path}")
    print(f"  saved {args.out_dir}/pooled{tag}.csv")


SUMMARY_ROWS = [
    ("LOSS", None, None),
    ("loss ratio",            "loss_ratio",            "{:.3e}"),
    ("vs fixed (x better)",   "_rel_fixed",            "{:s}"),
    ("dropped [Mb]",          "dropped_Mb",            "{:.1f}"),
    ("BACKLOG [Mb]", None, None),
    ("mean",                  "backlog_mean_Mb",       "{:.3f}"),
    ("sd",                    "backlog_sd_Mb",         "{:.3f}"),
    # ("p95",                   "backlog_p95_Mb",        "{:.3f}"),
    # ("p99",                   "backlog_p99_Mb",        "{:.3f}"),
    # ("max",                   "backlog_max_Mb",        "{:.3f}"),
    ("BEAM", None, None),
    ("served [Mb/s]",         "served_Mbps",           "{:.2f}"),
    ("wasted, avoidable [%]", "waste_avoidable_pct",   "{:.3f}"),
    ("wasted, system dry [%]", "waste_unavoidable_pct", "{:.3f}"),
    ("FORECAST", None, None),
    ("MASE",                  "forecast_mase",         "{:.4f}"),
    ("slot err [slots]",      "slot_err",              "{:.3f}"),
    ("deficit [Mb/period]",   "deficit_Mb_per_period", "{:.4f}"),
    ("surplus [Mb/period]",   "surplus_Mb_per_period", "{:.4f}"),
]


def _summary_rows(n_cells):
    """SUMMARY_ROWS with the per-cell backlog block spliced in after the
    aggregate one. Built at call time because the cell count is a run-time
    argument. The per-cell block is where two planners actually differ -- the
    aggregate above it is close to forecast-invariant."""
    out = []
    for entry in SUMMARY_ROWS:
        out.append(entry)
        if entry[1] == "backlog_sd_Mb":           # end of the aggregate block
            out.append(("PER-CELL BACKLOG [Mb]", None, None))
            for stat, fmt in (("mean", "{:.3f}"), ("max", "{:.3f}")):
                for i in range(n_cells):
                    out.append((f"{stat} c{i}", f"backlog_{stat}_c{i}_Mb", fmt))
            out.append(("worst cell max", "backlog_max_cell_Mb", "{:.3f}"))
            out.append(("imbalance (max-min mean)", "backlog_imbalance_Mb",
                        "{:.4f}"))
        if entry[1] == "forecast_mase":
            # The row above is scale-weighted across cells; these are not.
            for i in range(n_cells):
                out.append((f"MASE c{i}", f"forecast_mase_c{i}", "{:.4f}"))
            out.append(("MASE, equal-weighted", "forecast_mase_eq", "{:.4f}"))
            out.append(("MASE, worst cell", "forecast_mase_worst", "{:.4f}"))
            # std(f - mean f) / std(y - mean y): the share of the demand's
            # swing the forecast reproduces. 1 = matched amplitude.
            out.append(("sd ratio", "forecast_sd_ratio", "{:.4f}"))
            for i in range(n_cells):
                out.append((f"sd ratio c{i}", f"forecast_sd_ratio_c{i}",
                            "{:.4f}"))
            out.append(("sd ratio, equal-weighted", "forecast_sd_ratio_eq",
                        "{:.4f}"))
    return out


def write_summary(rows, sample_dt, args, tag, rep=0):
    """Per-granularity text dump for ONE replicate: the same numbers report()
    prints, kept next to the figures so a PNG never has to be read without its
    values. Across-replicate numbers live in summary_pooled.txt instead."""
    tg = f"{sample_dt:g}s".replace(".", "p")
    path = f"{args.out_dir}/summary_{tg}{tag}.txt"
    sub = [r for r in rows
           if r["sample_dt"] == sample_dt and r.get("replicate", 0) == rep]

    # Loss relative to the no-forecast floor -- the headline of the study, and
    # the one number that is painful to compute by eye from two exponentials.
    base = next((r["loss_ratio"] for r in sub if r["policy"] == "fixed"), None)
    for r in sub:
        if not base or not r["loss_ratio"] or r["policy"] == "fixed":
            r["_rel_fixed"] = "-"                 # fixed is the baseline itself
        else:
            r["_rel_fixed"] = f"{base / r['loss_ratio']:.1f}x"

    pols = [r["policy"] for r in sub]
    w = max(12, max(len(p) for p in pols) + 2)
    summary_rows = _summary_rows(args.n_cells)
    lab_w = max(len(l) for l, _, _ in summary_rows) + 1

    def cell(r, col, fmt):
        v = r.get(col)
        if v is None:
            return "-"
        if isinstance(v, str):
            return v
        return "nan" if v != v else fmt.format(v)

    with open(path, "w") as f:
        f.write(f"sample_dt = {sample_dt:g}s   {args.n_cells} cells / 1 beam   "
                f"alpha={args.alpha_str} "
                f"(H={'|'.join('%.3f' % ((3-a)/2) for a in args.alpha)})   "
                f"buffer {args.buffer_mb:g} MiB "
                f"({args.buffer_mb * MIB * 8 / 1e6:.2f} Mbit/cell)\n")
        f.write(f"offered {sub[0]['offered_Mbps']:.1f} Mb/s over "
                f"{sub[0]['beam_Mbps']:.0f} Mb/s beam -> "
                f"rho = {sub[0]['rho']:.4f}   "
                f"test {args.test_hours:g}h   "
                f"adaptive_mean={sub[0]['adaptive_mean']}\n")
        f.write(f"decisions = {sub[0]['n_decisions']}   "
                f"train = {sub[0]['train_seconds']:g}s "
                f"({args.train_samples} samples)\n")
        hm = [sub[0].get(f"H_meas_c{i}") for i in range(args.n_cells)]
        if all(h is not None for h in hm):
            f.write("H measured (local Whittle, test window) = "
                    + "|".join("%.3f" % h for h in hm) + "\n")
        f.write("=" * (lab_w + w * len(pols)) + "\n")
        f.write(" " * lab_w + "".join(p.rjust(w) for p in pols) + "\n")

        for label, col, fmt in summary_rows:
            if col is None:                       # section heading
                f.write("\n" + label + "\n")
                continue
            f.write("  " + label.ljust(lab_w - 2)
                    + "".join(cell(r, col, fmt).rjust(w) for r in sub) + "\n")

        dcols = [c for c in sub[0] if c.startswith("drop_c")]
        if dcols:
            f.write("\nPER-CELL DROPPED [Mb]\n")
            for c in dcols:
                f.write("  " + c.replace("drop_", "").replace("_Mb", "")
                        .ljust(lab_w - 2)
                        + "".join(f"{r[c]:.1f}".rjust(w) for r in sub) + "\n")
    print(f"  saved {path}")


def save_tables(rows, sweep_rows, args, tag):
    pd.DataFrame(rows).to_csv(f"{args.out_dir}/metrics{tag}.csv", index=False)
    print(f"  saved {args.out_dir}/metrics{tag}.csv ({len(rows)} rows)")
    if sweep_rows:
        pd.DataFrame(sweep_rows).to_csv(
            f"{args.out_dir}/loss_vs_buffer{tag}.csv", index=False)


def plot_backlogs(results, sample_dt, args, cap, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = f"{sample_dt:g}s".replace(".", "p")
    n = results[args.planners[0]]["occupancy"].shape[1]
    t = np.arange(n) * ARRIVAL_DT * args.trace_decim / 3600.0    # hours
    ymax = max(r["occupancy"].sum(axis=0).max() for r in results.values()) / 1e6

    for k, r in results.items():
        occ = r["occupancy"] / 1e6
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, occ.sum(axis=0), lw=0.4, color="k")
        axes[0].set_ylabel("total backlog [Mb]")
        axes[0].set_ylim(0, ymax * 1.05)
        for i in range(args.n_cells):
            axes[1].plot(t, occ[i], lw=0.4, label=f"cell{i}")
        if cap is not None:
            axes[0].axhline(args.n_cells * cap / 1e6, color="r", lw=0.8,
                            ls="--", label="aggregate capacity")
            axes[1].axhline(cap / 1e6, color="r", lw=0.8, ls="--",
                            label="per-cell capacity")
            axes[0].legend(fontsize=8)
        axes[1].set_ylabel("per-cell backlog [Mb]")
        axes[1].set_xlabel("time [h]")
        axes[1].legend(fontsize=8, ncol=4)
        row = [x for x in rows
               if x["policy"] == k and x["sample_dt"] == sample_dt
               and x.get("replicate", 0) == 0][-1]
        fig.suptitle(
            f"backlog: {k}  --  sample_dt={sample_dt:g}s, "
            f"{args.n_cells} cells / 1 beam, alpha={args.alpha_str}, "
            f"buffer {args.buffer_mb:g} MiB\n"
            f"mean {row['backlog_mean_Mb']:.3f} Mb   "
            f"max {row['backlog_max_Mb']:.3f} Mb   "
            f"loss {row['loss_ratio']:.3e}", fontsize=10)
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/backlog_{tag}_{k}.png", dpi=120)
        plt.close(fig)
    print(f"  saved {args.out_dir}/backlog_{tag}_*.png")


if __name__ == "__main__":
    main()
