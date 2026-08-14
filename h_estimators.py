import numpy as np
from scipy.optimize import minimize_scalar


def local_whittle(x, m=None, exclude=None):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if m is None:
        m = int(n ** 0.65)          # standard bandwidth choice
    m = max(8, min(m, n // 2 - 1))

    # periodogram at lambda_j = 2*pi*j/n, j = 1..m (mean removed => j=0 dropped)
    per = np.abs(np.fft.rfft(x - x.mean())) ** 2 / (2.0 * np.pi * n)
    j = np.arange(1, m + 1)
    I = per[1:m + 1]
    lam = 2.0 * np.pi * j / n

    if exclude is not None:
        keep = ~np.asarray(exclude(j) if callable(exclude) else exclude, bool)
        if keep.sum() < 8:
            raise ValueError(f"exclude left only {keep.sum()} of {m} bins")
        I, lam = I[keep], lam[keep]

    log_lam_bar = np.log(lam).mean()

    def R(d):
        return np.log((lam ** (2.0 * d) * I).mean()) - 2.0 * d * log_lam_bar

    res = minimize_scalar(R, bounds=(-0.499, 0.499), method="bounded")
    return res.x + 0.5


def fgn(n, H, rng):
    """Exact fractional Gaussian noise (Davies-Harte), for calibration."""
    k = np.arange(n)
    g = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H)
               + np.abs(k + 1) ** (2 * H))
    c = np.concatenate([g, [0.0], g[:0:-1]])
    lam = np.fft.fft(c).real
    lam[lam < 0] = 0.0
    m = len(c)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    return np.fft.fft(np.sqrt(lam / (2 * m)) * w).real[:n]


H_GRID = np.arange(0.55, 0.9951, 0.005)


def fit_H_forecast(Z, T, horizon, grid=H_GRID):

    from linearP_f import linearp_weights          # deferred: avoids a cycle
    from norros_f import design

    win, truth = design(np.asarray(Z, dtype=float), T, horizon)
    if len(truth) < 2:
        raise ValueError(f"need > T + horizon = {T + horizon} samples to fit "
                         f"H, got {len(Z)}")
    grid = np.asarray(grid, dtype=float)
    err = [np.mean((truth - win @ linearp_weights(T, H, horizon)) ** 2)
           for H in grid]
    return float(grid[int(np.argmin(err))])


def analyze_fit(T, horizon):

    def analyze(rate_data, dt):
        x = np.asarray(rate_data, dtype=float)
        H = fit_H_forecast((x - x.mean()) / x.std(), T, horizon)
        m = np.mean(x) / dt
        a = (np.std(x) / (m * dt ** H)) ** 2 * m
        return m, a, H
    return analyze
