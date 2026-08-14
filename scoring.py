import numpy as np


# Methods reported in the traffic-units table, and the columns written to
# --out. Adding or removing an entry here is the only edit needed; the report,
# the per-forecast dump and the --out schema all follow it.
RAW_METHODS = [
    ("gt", "norros_f"),
    ("lp", "linearP_f"),
    ("far", "farima_f"),
    ("naive", "persist"),
]


def naive_scale(truth):
    truth = np.asarray(truth, dtype=float)
    if truth.size < 2:
        raise ValueError("MASE needs at least 2 points to form a naive scale")
    return float(np.mean(np.abs(np.diff(truth))))


def raw_space_nmse(r, comp, xf, invert=None):

    T, h, b = r["T"], r["horizon"], r["test_start"]
    n_fc = r["n_fc"]

    # Cumulative sums make the h-bucket window sums a difference of two terms;
    # with h=1 this is just the bucket itself.
    cs = np.concatenate(([0.0], np.cumsum(xf)))
    lo = b + np.arange(n_fc) + T
    truth = cs[lo + h] - cs[lo]

    if invert is None:
        cc = np.concatenate(([0.0], np.cumsum(comp)))
        comp_sum = cc[lo + h] - cc[lo]

    denom = np.var(truth)
    scale = naive_scale(truth)
    metrics, preds = {}, {}
    for name, w in r["weights"].items():
        x_sum = r["z_scale"] * (r["win"] @ w) + h * r["z_offset"]
        p = invert(name, x_sum, lo, h) if invert else x_sum + comp_sum
        preds[name] = p
        se = (truth - p) ** 2
        metrics[name] = dict(mse=float(np.mean(se)),
                             nmse=float(np.mean(se) / denom),
                             mase=float(np.mean(np.abs(truth - p)) / scale))

    return metrics, lo, truth, preds


