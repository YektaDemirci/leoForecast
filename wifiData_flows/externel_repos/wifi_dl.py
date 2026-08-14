import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

from data_provider.data_loader import Dataset_Custom
from models import DLinear

SEQ_LEN = int(os.environ.get('SEQ_LEN', 48))
PRED_LEN = 1
ROOT = './'
RAW, DES = 'wifiData.csv', 'wifiData_deseasonalised.csv'
RES = './res.txt'
CKPT = './checkpoints/{}_DLinear_1.04_%d_%d_t/checkpoint.pth' % (SEQ_LEN, PRED_LEN)


class Cfg:
    seq_len, pred_len, individual, enc_in = SEQ_LEN, PRED_LEN, False, 1


def train(data_path):

    if int(os.environ.get('SKIP_TRAIN', 0)):
        if not os.path.exists(CKPT.format(data_path)):
            raise SystemExit('SKIP_TRAIN=1 but no checkpoint for %s' % data_path)
        print('SKIP_TRAIN=1: reusing existing checkpoint for %s' % data_path)
        return
    print('training %s ...' % data_path)
    subprocess.run([sys.executable, 'run_longExp.py',
                    '--is_training', '1', '--model_id', 'wifi', '--parr', '1.04',
                    '--root_path', ROOT, '--data_path', data_path,
                    '--model', 'DLinear', '--data', 'custom',
                    '--features', 'S', '--target', 'OT',
                    '--seq_len', str(SEQ_LEN), '--label_len', '0',
                    '--pred_len', str(PRED_LEN), '--enc_in', '1', '--freq', 't',
                    '--itr', '1',
                    '--learning_rate', '0.005', '--train_epochs', '40',
                    '--lradj', '3', '--patience', '15',
                    '--batch_size', '32'],
                   check=True, stdout=subprocess.DEVNULL)


def forecast(data_path):
    """Return (pred, true) 1-step forecasts on the test split, in the file's own units."""
    ds = Dataset_Custom(root_path=ROOT, flag='test', size=[SEQ_LEN, 0, PRED_LEN],
                        features='S', data_path=data_path, target='OT',
                        scale=True, timeenc=1, freq='t')
    model = DLinear.Model(Cfg())
    model.load_state_dict(torch.load(CKPT.format(data_path), map_location='cpu'))
    model.eval()

    x = np.stack([ds[i][0] for i in range(len(ds))])              # (N, seq_len, 1)
    with torch.no_grad():
        pred = model(torch.from_numpy(x).float()).numpy()[:, -1, 0]
    true = np.stack([ds[i][1] for i in range(len(ds))])[:, -1, 0]

    inv = lambda v: ds.inverse_transform(v.reshape(-1, 1)).ravel()
    return inv(pred), inv(true), ds


def seasonal_component(raw_path, des_path):

    raw = pd.read_csv(os.path.join(ROOT, raw_path), parse_dates=['date'])
    des = pd.read_csv(os.path.join(ROOT, des_path), parse_dates=['date'])
    raw = raw.sort_values('date').reset_index(drop=True)
    des = des.sort_values('date').reset_index(drop=True)
    if len(raw) != len(des) or not (raw.date.values == des.date.values).all():
        raise SystemExit('%s and %s are not aligned -- regenerate the '
                         'deseasonalised file with wifi_deseason.py'
                         % (raw_path, des_path))
    x = raw.OT.to_numpy(dtype=float)
    return x - des.OT.to_numpy(dtype=float), x


def dump(path, name, idx, truth, season, pred):

    date = pd.read_csv(os.path.join(ROOT, RAW), parse_dates=['date']) \
             .sort_values('date').reset_index(drop=True).date.to_numpy()[idx]
    pd.DataFrame({'idx': idx, 'date': date, 'truth': truth,
                  'season': season, name: pred}).to_csv(path, index=False)
    print('per-forecast values -> %s' % path)


def mase(pred, true):

    true = np.asarray(true, dtype=float)
    scale = np.mean(np.abs(np.diff(true)))
    return np.mean(np.abs(true - np.asarray(pred, dtype=float))) / scale


def append_res(series, value):
    """One line per series in res.txt, keeping the existing 5-field layout."""
    needs_nl = os.path.exists(RES) and os.path.getsize(RES) > 0
    if needs_nl:
        with open(RES, 'rb') as f:                    # previous runs left no trailing \n
            f.seek(-1, os.SEEK_END)
            needs_nl = f.read(1) != b'\n'
    with open(RES, 'a') as f:
        f.write(('\n' if needs_nl else '') +
                'DLINEAR %d %s %s %.4f\n' % (SEQ_LEN, RAW, series, value))


season, x_raw = seasonal_component(RAW, DES)

train(RAW)
train(DES)

p_raw, t_raw, ds_raw = forecast(RAW)
p_des, t_des, ds_des = forecast(DES)

# absolute row index of each test target: border1 + i + seq_len, border1 = n - num_test - seq_len
n = len(x_raw)
num_test = int(n * 0.2)
border1 = n - num_test - SEQ_LEN
idx = border1 + np.arange(len(p_des)) + SEQ_LEN
assert len(p_raw) == len(p_des)

p_des_re = p_des + season[idx]          # re-seasonalised forecast
truth = x_raw[idx]                      # raw ground truth

print(f'test windows: {len(idx)}  rows {idx[0]}..{idx[-1]}')

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
dump('./fc_dlinear_raw.csv', 'DLINEAR', idx, truth, np.zeros(len(idx)), p_raw)
dump('./fc_dlinear_des.csv', 'DLINEAR', idx, truth, season[idx], p_des_re)


mase_raw, mase_des = mase(p_raw, truth), mase(p_des_re, truth)
append_res('raw', mase_raw)
append_res('deseasonalized', mase_des)

print('--- MASE against RAW ground truth, original units ---')
print('raw model                     : %.4f' % mase_raw)
print('deseasonalised model + season : %.4f' % mase_des)
# Forecasting des = 0, i.e. the removed component on its own. The old
# transform added the train mean back into des, so this line used to add it
