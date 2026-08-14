# Yekta comment, sns-3 makes a cold start causing a higher variance and a value!
# In this version we make some of them ON and OFF, not a sharp swing in that sense.

import pandas as pd
import torch


# ----------------------------------------------------------------------------
# Sampling
# ----------------------------------------------------------------------------

def rand64(shape, device):
    if device.type != "cuda":
        return torch.rand(shape, device=device, dtype=torch.float64)
    hi = torch.rand(shape, device=device, dtype=torch.float32).double()
    lo = torch.rand(shape, device=device, dtype=torch.float32).double()
    return hi.add_(lo, alpha=2.0 ** -24).clamp_min_(2.0 ** -49)


def pareto(shape, alpha, xm, device):
    # Inverse-CDF sampling: X = xm * U^(-1/alpha), support [xm, inf)
    u = rand64(shape, device)
    return xm * u.pow(-1.0 / alpha)


def pareto_residual(shape, alpha, xm, device):
    # Equilibrium (residual-life) distribution of Pareto(alpha, xm),
    # f_e(x) = P(X > x) / E[X]. Needed to start sources in steady state.
    # Piecewise inverse CDF: linear on [0, xm], power-law (index alpha-1) above.
    mu = alpha * xm / (alpha - 1.0)
    u = rand64(shape, device)
    x_lin = u * mu
    x_tail = ((1.0 - u) * (alpha - 1.0) * mu / xm ** alpha) ** (1.0 / (1.0 - alpha))
    return torch.where(u <= (alpha - 1.0) / alpha, x_lin, x_tail)


# ----------------------------------------------------------------------------
# Simulation
# ----------------------------------------------------------------------------

def _bin_on_intervals(starts, ends, n_bins, dt, occ, diff):
    T = n_bins * dt
    s = starts.clamp(0.0, T)
    e = ends.clamp(0.0, T)
    keep = e > s
    s, e = s[keep], e[keep]

    i0 = torch.clamp((s / dt).long(), max=n_bins - 1)
    i1 = torch.clamp((e / dt).long(), max=n_bins)

    diff.zero_()

    same = i0 == i1
    occ.scatter_add_(0, i0[same], (e - s)[same])

    multi = ~same
    im0, im1 = i0[multi], i1[multi]
    sm, em = s[multi], e[multi]
    occ.scatter_add_(0, im0, (im0 + 1).double() * dt - sm)   # head fragment
    occ.scatter_add_(0, im1, em - im1.double() * dt)          # tail fragment
    full = torch.full_like(sm, dt)
    diff.scatter_add_(0, im0 + 1, full)                       # middle bins,
    diff.scatter_add_(0, im1, -full)                          # range update

    occ.add_(diff.cumsum_(0))   # in place: no second full-length allocation


def simulate_onoff_aggregate(n_sources, T, dt, alpha_on, alpha_off,
                             xm_on=1.0, xm_off=1.0, rate_on=1.0,
                             device=None, source_batch=None, seed=None,
                             mem_budget_bytes=500e6, stationary_start=True):


    if not (1.0 < alpha_on < 2.0 and 1.0 < alpha_off < 2.0):
        raise ValueError("alpha_on/alpha_off must lie in (1, 2) for the "
                         "heavy-tailed self-similar regime")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    mu_on = alpha_on * xm_on / (alpha_on - 1.0)
    mu_off = alpha_off * xm_off / (alpha_off - 1.0)
    p_on = mu_on / (mu_on + mu_off)

    n_bins = int(round(T / dt))
    # length n_bins + 1: _bin_on_intervals uses the extra slot as a sink for
    # intervals that reach the horizon. Allocated once and reused by every
    # source batch; the trailing bin is dropped on return.
    on_time = torch.zeros(n_bins + 1, device=device, dtype=torch.float64)
    diff = torch.zeros(n_bins + 1, device=device, dtype=torch.float64)

    if source_batch is None:
        source_batch = n_sources

    for lo in range(0, n_sources, source_batch):
        B = min(source_batch, n_sources - lo)

        # Cycles are generated in chunks of K and binned as we go, carrying each
        # source's clock forward in `t0`, rather than materialising every cycle
        # up to T at once. The horizon then costs no memory: peak usage is set
        # by K alone. This is not a detail at alpha near 1 -- the number of
        # cycles to cover T is itself heavy-tailed, so no a-priori K is safe,
        # and the old resample-with-K*=2 fallback doubled the allocation until
        # it hit the card. ~8 float64 tensors of shape (B, 2K) live at once.
        K = max(64, int(mem_budget_bytes / (B * 2 * 8 * 8)))

        idx = torch.arange(B, device=device)   # sources still short of T
        t0 = torch.zeros(B, device=device, dtype=torch.float64)
        first = True
        while idx.numel():
            b = idx.numel()
            # Layout per source: [off_0, on_0, off_1, on_1, ..., off_K, on_K].
            offs = pareto((b, K), alpha_off, xm_off, device)
            ons = pareto((b, K), alpha_on, xm_on, device)
            if first and stationary_start:
                # ON-starters get a zero-length off_0 and a residual first ON;
                # OFF-starters get a residual first OFF. Only the chunk that
                # starts at t=0 gets this: later chunks resume mid-stream,
                # where an ordinary fresh Pareto draw is already correct.
                start_on = rand64((b,), device) < p_on
                offs[:, 0] = torch.where(
                    start_on, torch.zeros_like(offs[:, 0]),
                    pareto_residual((b,), alpha_off, xm_off, device))
                ons[:, 0] = torch.where(
                    start_on, pareto_residual((b,), alpha_on, xm_on, device),
                    ons[:, 0])
            # else: leave off_0/on_0 as ordinary Pareto draws -- every source
            # starts OFF at t=0 and waits out a full fresh OFF period.
            first = False

            durations = torch.stack([offs, ons], dim=2).reshape(b, -1)
            del offs, ons
            # edges[:, 2k] opens on_k, edges[:, 2k+1] closes it. The chunk
            # starts at each source's own t0, so cycles chain across chunks
            # exactly as they would have in one long cumsum.
            edges = torch.cumsum(durations, dim=1).add_(t0[idx].unsqueeze(1))
            del durations

            on_starts = edges[:, 0::2].reshape(-1)
            on_ends = edges[:, 1::2].reshape(-1)
            # Intervals past T are clamped to zero length and dropped there.
            _bin_on_intervals(on_starts, on_ends, n_bins, dt, on_time, diff)

            t0[idx] = edges[:, -1]
            del edges, on_starts, on_ends
            idx = idx[t0[idx] < T]   # drop sources that have reached T

    del diff
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # aggregate rate averaged over each bin; in place, the buffer is not reused
    rate = on_time.mul_(rate_on / dt)[:n_bins]
    return rate, p_on


# ----------------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------------

def estimate_hurst(series, min_scale_bins=1):
    # Variance-time plot: Var(sum of k consecutive bins) ~ k^{2H}.
    # Scales below the Pareto cutoff xm show near-deterministic slope ~2,
    # so start the fit at min_scale_bins (a few xm/dt) to avoid that bias.
    x = np.asarray(series, dtype=np.float64)
    x = x - x.mean()
    ks, vs = [], []
    k = max(1, int(min_scale_bins))
    while len(x) // k >= 50:
        n_blocks = len(x) // k
        sums = x[:n_blocks * k].reshape(n_blocks, k).sum(axis=1)
        ks.append(k)
        vs.append(sums.var())
        k = max(k + 1, int(k * 1.3))
    slope = np.polyfit(np.log(ks), np.log(vs), 1)[0]
    return slope / 2.0


if __name__ == "__main__":
    # Parameters
    n_sources = 500     # X sources
    alpha_on = 1.04       # Pareto tail index of ON durations
    alpha_off = 1.04      # Pareto tail index of OFF durations
    xm_on = 0.001          # minimum ON duration [s]
    xm_off = 0.001         # minimum OFF duration [s]
    rate_on = 1e6        # emission rate of one ON source [units/s]
    dt = 0.1             # bin width [s]
    T = 1000.0          # horizon [s]  ->  T/dt bins
    # False = ns-3 OnOffApplication behaviour, matching f_sns100.csv. See the
    # simulate_onoff_aggregate docstring; this is worth ~3-4x in the Norros `a`.
    stationary_start = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H_theory = (3.0 - min(alpha_on, alpha_off)) / 2.0
    print(f"device={device}  X={n_sources}  alpha=({alpha_on},{alpha_off})  "
          f"theoretical H={H_theory:.3f}  "
          f"start={'stationary' if stationary_start else 'cold (ns-3)'}")

    rate, p_on = simulate_onoff_aggregate(
        n_sources, T, dt, alpha_on, alpha_off, xm_on, xm_off, rate_on,
        device=device, seed=42, stationary_start=stationary_start)
    rate_np = rate.cpu().numpy()

    m_theory = n_sources * rate_on * p_on
    print(f"mean rate: empirical {rate_np.mean():.2f}  "
          f"theory X*r*p_on = {m_theory:.2f}")
    H_hat = estimate_hurst(rate_np, min_scale_bins=5 * max(xm_on, xm_off) / dt)
    print(f"Hurst:     estimated {H_hat:.3f}  theory {H_theory:.3f} "
          f"(variance-time converges slowly for alpha near 2)")

    # Save in the same (date, OT) layout as the other datasets. Those store
    # work *per bin* (f_sns100.csv: mean 2.55e7 per 0.1 s bin = 2.55e8 b/s),
    # so multiply the rate back by dt -- downstream code recovers the rate as
    # mean(OT)/dt and would otherwise divide by dt a second time.
    idx = pd.date_range("2025-01-01", periods=len(rate_np),
                        freq=f"{int(dt * 1000)}ms")
    pd.DataFrame({"date": idx, "OT": rate_np * dt}).to_csv(
        "./data/ETT/onoff_pareto4.csv", index=False)
    print("saved ./data/ETT/onoff_pareto3.csv")

    # Visual self-similarity check: the trace at three aggregation levels
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    for ax, agg in zip(axes, [1, 10, 100]):
        n = len(rate_np) // agg * agg
        y = rate_np[:n].reshape(-1, agg).mean(axis=1)
        ax.plot(np.arange(len(y)) * dt * agg, y, lw=0.5)
        ax.set_ylabel(f"rate (agg x{agg})")
    axes[-1].set_xlabel("time [s]")
    fig.suptitle(f"Aggregate of {n_sources} Pareto ON/OFF sources "
                 f"(H_theory={H_theory:.2f})")
    fig.tight_layout()
    fig.savefig("onoff_pareto_traffic.png", dpi=120)
    print("saved onoff_pareto_traffic.png")
