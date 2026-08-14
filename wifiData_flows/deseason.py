"""The deseasonalisation, shared by the plot script and the forecasting flow.
"""

import numpy as np
import pandas as pd

SLOTS_PER_DAY = 144          # 10-minute buckets

MEAN_DAYS = 7

SMOOTH_SLOTS = 11


def rolling_slot_mean(y, tod, n_fit, days=MEAN_DAYS):
    """Per-slot trailing mean over the previous `days` days.

    groupby(tod).rolling makes the window `days` samples of the SAME 10-minute
    slot, i.e. a window of days; shift(1) makes it causal. Warm-up, where fewer
    than 2 prior days exist, falls back to the train-fitted fixed profile.
    """
    prof = y.groupby(tod).transform(
        lambda v: v.shift(1).rolling(days, min_periods=2).mean())
    fixed = tod.map(y[:n_fit].groupby(tod[:n_fit]).mean())
    return prof.fillna(fixed)


def circ_smooth(vals, k=SMOOTH_SLOTS):
    v = np.asarray(vals, dtype=float)
    if k <= 1 or v.size < 2:
        return v.copy()
    k = min(int(k) | 1, v.size)
    w = np.ones(k) / k
    pad = k // 2
    return np.convolve(np.r_[v[-pad:], v, v[:pad]], w, mode="valid")


def _smooth_across_tod(vals, tod, group, k=SMOOTH_SLOTS):
    if k <= 1:
        return np.asarray(vals, dtype=float)
    s = pd.Series(np.asarray(vals, dtype=float)).reset_index(drop=True)
    g = pd.Series(np.asarray(group)).reset_index(drop=True)
    t = pd.Series(np.asarray(tod)).reset_index(drop=True)
    out = s.copy()
    for _key, sub in s.groupby(g, sort=False):
        order = t.loc[sub.index].sort_values().index
        out.loc[order] = circ_smooth(s.loc[order].to_numpy(), k)
    return out.to_numpy()


def deseasonalise_parts(df, n_fit, col="OT"):
    tod = df.date.dt.hour * 6 + df.date.dt.minute // 10
    tow = df.date.dt.dayofweek * SLOTS_PER_DAY + tod
    x = df[col].astype(float)

    tr = slice(0, n_fit)
    mu = rolling_slot_mean(x, tod, n_fit)
    mu = pd.Series(_smooth_across_tod(mu, tod, df.date.dt.normalize()),
                   index=x.index)
    z = x - mu

    prof_week = z[tr].groupby(tow[tr]).mean()
    prof_week = prof_week.reindex(range(7 * SLOTS_PER_DAY)).fillna(0.0)
    pw_tod = prof_week.index % SLOTS_PER_DAY
    pw_dow = prof_week.index // SLOTS_PER_DAY
    prof_week = pd.Series(
        _smooth_across_tod(prof_week.to_numpy(), pw_tod, pw_dow),
        index=prof_week.index)
    wk = tow.map(prof_week).fillna(0.0)
    z = z - wk

    comp = np.asarray(mu + wk, dtype=float)
    return np.asarray(z, dtype=float), np.asarray(x, dtype=float), comp


def deseasonalise(df, n_fit, col="OT"):
    """(des, x) only -- the signature plot_deseasonalizeWifi.py uses."""
    des, x, _comp = deseasonalise_parts(df, n_fit, col)
    return des, x
