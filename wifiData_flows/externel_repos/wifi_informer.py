import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.data_loader import Dataset_Custom
from exp.exp_informer import Exp_Informer

SEQ_LEN = int(os.environ.get('SEQ_LEN', 48))
LABEL_LEN = int(os.environ.get('LABEL_LEN', 24))
PRED_LEN = 1
EPOCHS = int(os.environ.get('EPOCHS', 10))
SEED = int(os.environ.get('SEED', 2021))
SKIP_TRAIN = os.environ.get('SKIP_TRAIN', '0') == '1'
ROOT = './dataset/'
RAW, DES = 'wifiData.csv', 'wifiData_deseasonalised.csv'
RES = './res.txt'
FREQ = 't'                       # 10-minute buckets -> minute-level features


def build_args(data_path, tag):
    """The subset of main_informer.py's namespace that Exp_Informer reads."""
    a = argparse.Namespace(
        model='informer', data='custom', root_path=ROOT, data_path=data_path,
        features='S', target='OT', freq=FREQ, detail_freq=FREQ,
        checkpoints='./checkpoints/',
        seq_len=SEQ_LEN, label_len=LABEL_LEN, pred_len=PRED_LEN,
        enc_in=1, dec_in=1, c_out=1,
        d_model=512, n_heads=8, e_layers=2, d_layers=1, s_layers=[3, 2, 1],
        d_ff=2048, factor=5, padding=0, distil=True, dropout=0.05,
        attn='prob', embed='timeF', activation='gelu', output_attention=False,
        do_predict=False, mix=True, cols=None,
        num_workers=0, itr=1, train_epochs=EPOCHS, batch_size=32, patience=3,
        learning_rate=1e-4, des='Exp', loss='mse', lradj='type1',
        use_amp=False, inverse=False,
        use_gpu=torch.cuda.is_available(), gpu=0, use_multi_gpu=False,
        devices='0', seed=SEED, tag=tag,
    )
    return a


def forecast(data_path, tag):
    """Train (unless SKIP_TRAIN) and return 1-step test forecasts in file units."""
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    args = build_args(data_path, tag)
    setting = '{}_{}_{}_{}_{}'.format(data_path, SEQ_LEN, PRED_LEN, FREQ, tag)
    exp = Exp_Informer(args)

    ckpt = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    if SKIP_TRAIN and os.path.exists(ckpt):
        print(f'[{tag}] reusing {ckpt}')
    else:
        print(f'[{tag}] training on {data_path} ...')
        exp.train(setting)
    exp.model.load_state_dict(torch.load(ckpt))
    exp.model.eval()

    ds = Dataset_Custom(root_path=ROOT, flag='test',
                        size=[SEQ_LEN, LABEL_LEN, PRED_LEN],
                        features='S', data_path=data_path, target='OT',
                        scale=True, inverse=False, timeenc=1, freq=FREQ)
    loader = DataLoader(ds, batch_size=64, shuffle=False, drop_last=False)

    preds, trues = [], []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            out, y = exp._process_one_batch(ds, bx, by, bxm, bym)
            preds.append(out.detach().cpu().numpy())
            trues.append(y.detach().cpu().numpy())

    # (N, pred_len, 1) -> (N,), pred_len == 1
    pred = np.concatenate(preds)[:, -1, 0]
    true = np.concatenate(trues)[:, -1, 0]

    inv = lambda v: ds.scaler.inverse_transform(v.reshape(-1, 1)).ravel()
    return inv(pred), inv(true)


def seasonal_component():
    """The component the deseasonalisation removed, per absolute row index.

    Returns (comp, x_raw) with  x_raw == des + comp  exactly.

    Taken as raw - deseasonalised straight from the two files, so it is exact by
    construction rather than a re-fit that could drift from what was written.
    That is what kept this script correct when the transform changed, while the
    hardcoded re-fits in dl_wifi.py and chronos_wifi.py went silently wrong by
    ~45% of peak traffic; both now do it this way too. It does assume DES was
    regenerated from the current RAW by plot_deseasonalizeWifi.py, which the
    date alignment check enforces.
    """
    raw = pd.read_csv(os.path.join(ROOT, RAW), parse_dates=['date'])
    des = pd.read_csv(os.path.join(ROOT, DES), parse_dates=['date'])
    raw = raw.sort_values('date').reset_index(drop=True)
    des = des.sort_values('date').reset_index(drop=True)
    if len(raw) != len(des) or not (raw.date.values == des.date.values).all():
        raise SystemExit('raw and deseasonalised files are not aligned -- regenerate '
                         'the deseasonalised file with plot_deseasonalizeWifi.py')
    x = raw.OT.to_numpy(dtype=float)
    return x - des.OT.to_numpy(dtype=float), x


def dump(path, name, idx, truth, season, pred):
    """Write this run's forecasts for wifiData_flows/compare_forecasts.py.

    The forecasts, not a score. Three scripts in three repos each scoring
    themselves means three copies of the metric and three choices of test
    window, and both drifted: none of them wrote MSE, and they scored buckets
    7143..8927 while score_wifi.py scored 7190..8927. Emitting the forecasts
    and letting one script score every run on the intersection of the windows
    removes both problems, and this file no longer has to agree with anything.

    `season` is the component ADDED BACK -- zero for the raw-series run. It is
    what lets compare_forecasts.py verify the re-seasonalisation against
    deseason.py: a stale deseasonalised csv keeps its dates and row count, so
    the alignment assert above passes straight through one, and `truth` is the
    raw series in both variants so it agrees too.
    """
    date = pd.read_csv(os.path.join(ROOT, RAW), parse_dates=['date']) \
             .sort_values('date').reset_index(drop=True).date.to_numpy()[idx]
    pd.DataFrame({'idx': idx, 'date': date, 'truth': truth,
                  'season': season, name: pred}).to_csv(path, index=False)
    print('per-forecast values -> %s' % path)


def mase(pred, true):
    """MASE = MAE / mean absolute one-step change of `true`.

        MASE = [1/N sum |y - yhat|] / [1/(T-1) sum_{t=2..T} |y_t - y_{t-1}|]

    The naive scale is taken on the series being scored (the test-window truth),
    so 1.0 is exactly the one-step persistence forecast and the number is
    dimensionless -- the same convention scoring.py uses for the analytic
    predictors, so the two tables stay comparable.
    """
    true = np.asarray(true, dtype=float)
    scale = np.mean(np.abs(np.diff(true)))
    return np.mean(np.abs(true - np.asarray(pred, dtype=float))) / scale


def append_res(series, value):
    """One line per series in res.txt, keeping the existing 5-field layout."""
    needs_nl = os.path.exists(RES) and os.path.getsize(RES) > 0
    if needs_nl:
        with open(RES, 'rb') as f:                # previous runs left no trailing \n
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b'\n'
    with open(RES, 'a') as f:
        f.write(('\n' if needs_nl else '') +
                'INFORMER %d %s %s %.4f\n' % (SEQ_LEN, RAW, series, value))


season, x_raw = seasonal_component()

p_raw, t_raw = forecast(RAW, 'raw')
p_des, t_des = forecast(DES, 'deseas')

# Absolute row index of each test target. Dataset_Custom's test split starts at
# border1 = n - num_test - seq_len, and sample i targets border1 + i + seq_len.
n = len(x_raw)
num_test = int(n * 0.2)
border1 = n - num_test - SEQ_LEN
idx = border1 + np.arange(len(p_des)) + SEQ_LEN
assert len(p_raw) == len(p_des)

p_des_re = p_des + season[idx]          # re-seasonalised forecast
truth = x_raw[idx]                      # raw ground truth

print(f'\ntest windows: {len(idx)}  rows {idx[0]}..{idx[-1]}')
# comp is exact by construction, so this checks the INDEXING -- that idx really
# is the absolute row each Dataset_Custom test sample targets. A misaligned idx
# is the one bug the subtraction cannot rule out, and it would quietly inflate
# both MASEs, so it fails loudly instead of printing.
for _name, _err in (('des', np.abs(t_des + season[idx] - truth).max()),
                    ('raw', np.abs(t_raw - truth).max())):
    _rel = _err / np.abs(truth).max()
    print('sanity |true_%s - raw|max = %.3g  (relative %.2g)' % (_name, _err, _rel))
    # 1e-4, not tighter: the scalers round-trip through float32, so an exact
    # match is ~1e-7 relative at best. A genuinely misaligned idx is off by
    # O(0.1-1) relative, so this still catches it by orders of magnitude.
    if _rel > 1e-4:
        raise SystemExit('test-window indexing is off: reconstructed %s truth does '
                         'not match the raw series at idx' % _name)
print()
dump('./fc_informer_raw.csv', 'INFORMER', idx, truth, np.zeros(len(idx)), p_raw)
dump('./fc_informer_des.csv', 'INFORMER', idx, truth, season[idx], p_des_re)

# Kept alongside the dump, not replaced by it: these print/append the same
# numbers this script has always reported, so an existing res.txt stays
# readable. The authoritative table is now compare_forecasts.py, which scores
# the dumps above -- it uses one window and one denominator for every
# forecaster, and it reports MSE, which res.txt has no column for.
mase_raw, mase_des = mase(p_raw, truth), mase(p_des_re, truth)
append_res('raw', mase_raw)
append_res('deseasonalized', mase_des)

print('--- MASE against RAW ground truth, original units ---')
print('raw model                     : %.4f' % mase_raw)
print('deseasonalised model + season : %.4f' % mase_des)
# Forecasting des = 0, i.e. the removed component on its own. The old
# transform added the train mean back into des, so this line used to add it
# here too; the current one leaves des zero-mean, so comp alone is the baseline.
print('seasonal component alone      : %.4f' % mase(season[idx], truth))
print('persistence (y[t-1])          : %.4f' % mase(x_raw[idx - 1], truth))
print()
print('--- for reference, MASE in each model\'s own target space ---')
print('raw model  vs raw target      : %.4f' % mase(p_raw, t_raw))
print('des model  vs des target      : %.4f' % mase(p_des, t_des))
