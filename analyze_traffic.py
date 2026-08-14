"""Norros (m, a, H) traffic parameters, with H from the variance-time method.

The variance-time estimator is the one plotH.py draws, so the figure and the
forecasts now use the same H by construction.
"""
import numpy as np


def variance_by_scale(x, min_scale_bins=1, min_blocks=50, scales=None):
    """Variance of block SUMS at each aggregation scale.

    Returns (ks, variances). This is the curve behind the variance-time plot:
    Var(sum of k consecutive bins) ~ k**(2H), so the log-log slope is 2H.

    Scales stop once fewer than `min_blocks` blocks remain, since the variance
    of a handful of blocks is noise. Pass `scales` to evaluate on a given
    ladder instead of the default geometric one.

    The conventional "aggregated variance" plot uses block MEANS instead, where
    Var ~ m**(2H-2); the two differ by a factor k**2 and give the same H.
    """
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    ks, vs = [], []
    if scales is None:
        k = max(1, int(min_scale_bins))
        scales = []
        while len(x) // k >= min_blocks:
            scales.append(k)
            k = max(k + 1, int(k * 1.3))
    for k in scales:
        k = int(k)
        n_blocks = len(x) // k
        if n_blocks < 2:
            continue
        sums = x[:n_blocks * k].reshape(n_blocks, k).sum(axis=1)
        ks.append(k)
        vs.append(sums.var())
    return np.array(ks, dtype=float), np.array(vs, dtype=float)


def variance_time_H(x, min_scale_bins=1):
    """Aggregated-variance (variance-time plot) estimate of H.

    The log-log slope of variance_by_scale, halved. Note the fitting window:
    this starts at k = 1 and runs to len(x)//50. plotH.py fits from a larger
    n_min, and on this trace that alone moves H by more than 0.1 -- the curve
    is not straight, so the number depends on which scales are included.
    """
    ks, vs = variance_by_scale(x, min_scale_bins)
    slope = np.polyfit(np.log(ks), np.log(vs), 1)[0]
    return float(np.clip(slope / 2.0, 0.5001, 0.9999))


def analyze_traffic(rate_data, dt):
    """Norros (m, a, H) from the increments.

    m is the mean rate; a is the peakedness, defined so that
    Var(X(t)) = a * m * t**(2H) -- see the standardisation in
    single_cell_1s.run_one, Z = (x - m*dt) / sqrt(m*a).
    """
    increments = np.asarray(rate_data, dtype=float)
    H_hat = variance_time_H(increments)
    m_hat = np.mean(increments) / dt
    a2_hat = np.std(increments) / (m_hat * dt ** H_hat)
    a_hat = (a2_hat ** 2) * m_hat
    return m_hat, a_hat, H_hat


# Old names, kept so nothing that imported them breaks. All the same function
# now that only the variance-time estimator remains.
analyze_varscale = analyze_traffic
analyze_traffic_model = analyze_traffic
