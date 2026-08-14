import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import warnings

warnings.simplefilter("ignore", FutureWarning)

SLOTS_PER_DAY = 144          # 10-minute buckets
TRAIN_FRAC, VAL_FRAC = 0.70, 0.10

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from h_estimators import local_whittle   # noqa: E402  (needs ROOT on sys.path)
# The transform itself lives in deseason.py so score_wifi.py can import it
# without running this script; see that module's docstring for what it does and
# why. This script and the forecasting flow build the same series.
from deseason import deseasonalise   # noqa: E402

CSV = os.path.join(HERE, "wifiData.csv")
OUT_CSV = os.path.join(HERE, "wifiData_deseasonalised.csv")
OUT_PNG = os.path.join(HERE, "wifiData_deseasonalised.png")
RAW_PNG = os.path.join(HERE, "wifiData.png")   # was plotWifiData.py
COL = "OT"


# --- wavelet (Abry-Veitch) band slopes, same D4 as the rolling script -------
D4 = np.array([1 + np.sqrt(3), 3 + np.sqrt(3),
               3 - np.sqrt(3), 1 - np.sqrt(3)]) / (4 * np.sqrt(2))


def logscale(y, filt=D4):
    """(scale j, log2 mean detail^2, count) per dyadic scale; 2^j bins."""
    g = filt[::-1] * ((-1) ** np.arange(len(filt)))
    c = np.asarray(y, float) - np.mean(y)
    out, j = [], 1
    while len(c) >= 32:
        d = np.convolve(c, g, 'valid')[::2]
        c = np.convolve(c, filt, 'valid')[::2]
        if len(d) >= 8:
            out.append((j, np.log2(np.mean(d ** 2)), len(d)))
        j += 1
    return out


def wavelet_H(y, j_lo, j_hi):
    """Weighted LS slope of the logscale diagram over scales [j_lo, j_hi]."""
    rows = logscale(y)
    j = np.array([r[0] for r in rows], float)
    v = np.array([r[1] for r in rows], float)
    w = np.array([r[2] for r in rows], float)
    s = (j >= j_lo) & (j <= j_hi)
    if s.sum() < 3:
        raise ValueError(f"need >= 3 scales in [{j_lo}, {j_hi}], got {s.sum()}")
    A = np.vstack([j[s], np.ones(int(s.sum()))]).T
    W = np.diag(w[s])
    return (np.linalg.solve(A.T @ W @ A, A.T @ W @ v[s])[0] + 1) / 2


df = pd.read_csv(CSV, parse_dates=["date"]).sort_values("date").reset_index(drop=True)
n = len(df)
n_train = int(n * TRAIN_FRAC)
n_val = int(n * (TRAIN_FRAC + VAL_FRAC))

# One series throughout: the causal transform, profiles fitted on train only.
# The full-sample fit deseasonalise(df, n) used to be kept alongside it and was
# what every statistic below was measured on. It is gone deliberately -- having
# the CSV and the plot describe one series while the headline H described
# another meant the reported number was not a statistic of the data anyone
# forecasts. At matched settings the two differ by 0.013 (0.847 -> 0.834), so
# nothing is lost by dropping it; the leak in the profiles was never the thing
# moving H.
des, x = deseasonalise(df, n_train)

out = pd.DataFrame({"date": df.date, "OT": des})
out.to_csv(OUT_CSV, index=False)

# ---- stationarity ----------------------------------------------------------
from statsmodels.tsa.stattools import adfuller, kpss  # noqa: E402
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    adf_p = adfuller(des, autolag="AIC")[1]
    kpss_p = kpss(des, regression="c", nlags="auto")[1]
lag = SLOTS_PER_DAY
rho = np.corrcoef(des[:-lag], des[lag:])[0, 1]
print(f"KPSS p = {kpss_p:.3f} (stationarity passes)")

# ---- H, band by band -------------------------------------------------------
# Same harmonic-bin exclusion as deseasonalizeWifi.py.
J_DAY = n / SLOTS_PER_DAY
J_WEEK = J_DAY / 7.0
HARM_WIDTH = 3


def seasonal_bins(j):
    mask = np.zeros(j.shape, bool)
    for j0 in (J_DAY, J_WEEK):
        k = np.maximum(np.round(j / j0), 1.0)
        mask |= np.abs(j - k * j0) <= HARM_WIDTH
    return mask


# Headline: band-limited local Whittle. m = 94 keeps periods >= ~16 h, i.e.
# the band where the spectrum actually follows one power law (the docstring
# says why sub-daily frequencies must stay out). se = 1/(2 sqrt(m)) is
# Robinson's asymptotic standard error.
M_HEAD = 94
h_head = local_whittle(des, m=M_HEAD, exclude=seasonal_bins)
se = 1.0 / (2.0 * np.sqrt(M_HEAD))
print(f" Hurst value estimation via whittle H = {h_head:.2f} +/- {1.96*se:.2f}")

# Aggregating to 2 h bins pushes the steep intra-day band out of the
# estimator's bandwidth (self-similarity preserves H under aggregation), so
# the sweep should sit in 0.77-0.85 instead of climbing to the bound.
agg = des[:n // 12 * 12].reshape(-1, 12).mean(axis=1)
na = len(agg)
J_DAY_A, J_WEEK_A = na / 12.0, na / 84.0


def seasonal_bins_agg(j, width=2):
    mask = np.zeros(j.shape, bool)
    for j0 in (J_DAY_A, J_WEEK_A):
        k = np.maximum(np.round(j / j0), 1.0)
        mask |= np.abs(j - k * j0) <= width
    return mask


sweep = [(m, local_whittle(agg, m=m, exclude=seasonal_bins_agg))
         for m in (int(na ** e) for e in (0.5, 0.55, 0.6, 0.65, 0.7))]


# ---- plots, same style as the other two scripts -----------------------------
N_TICKS = 8


def human_bits(x, _pos=None):
    """Format y-axis tick values as human-readable bit counts, signed."""
    sign = '-' if x < 0 else ''
    x = abs(x)
    for unit, suffix in [(1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')]:
        if x >= unit:
            return f'{sign}{x/unit:.1f}{suffix}'
    return f'{sign}{int(x)}'


def panel(y, ylabel, out, yfmt=None):
    """One full-width time-series panel, saved to `out`."""
    fig, ax = plt.subplots(1, 1, figsize=(14, 4.5))
    fig.patch.set_facecolor('white')

    ax.set_facecolor('white')
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#cccccc')
    ax.tick_params(colors='#333333', labelsize=12)
    ax.yaxis.label.set_color('#333333')
    ax.grid(axis='y', color='#e0e0e0', linewidth=0.7, zorder=0)

    ax.fill_between(df.date, y, alpha=0.15, color='#378ADD')
    ax.plot(df.date, y, color='#378ADD', linewidth=1.2, zorder=3)
    ax.axhline(0, color='#cccccc', linewidth=0.7, zorder=1)
    if yfmt is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(yfmt))

    ax.set_ylabel(ylabel, fontsize=12)
    ax.margins(x=0)

    # evenly spaced ticks anchored on the first and last timestamps plotted
    first = matplotlib.dates.date2num(df.date.iloc[0])
    last = matplotlib.dates.date2num(df.date.iloc[-1])
    ax.set_xticks(np.linspace(first, last, N_TICKS))
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y-%m-%d'))

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', fontsize=12)

    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"plot -> {out}")


print()
panel(x, 'Traffic (bits)', RAW_PNG, yfmt=human_bits)
panel(des, 'Traffic residual (bits)', OUT_PNG, yfmt=human_bits)
print(f"data -> {OUT_CSV}")
