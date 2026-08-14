# leoForecast

The code-base used for the "Is Forecasting Accuracy Enough? A Comparative Study of Traffic Forecasters for Beam-Hopping LEO Satellite Networks" paper.

- `norros_f.py` — analytic Norros/Gripenberg fBm kernel `g_T`, integrated over unit cells (1 parameter: H)
- `linearP_f.py` — exactly optimal discrete linear predictor for fGn (same H, no discretization loss)
- `farima_f.py` — ARFIMA(p, d, 0) fitted once by MLE on the training block

## Layout

| Path | What it is |
| `syntheticData_flows/` | Synthetic Pareto ON/OFF generation and NMSE scoring |
| `wifiData_flows/` | CRAWDAD WiFi trace: parsing, deseasonalising, scoring |
| `beamHop/` | Beam-hopping simulation — 3 cells, 1 beam, loss ratio vs. forecast quality |

## Setup

```bash
python -m venv vv
source vv/bin/activate
pip install -r req.txt
```

## Run

Each `*_flows/` directory has its own `*.sh` runner for the full sweep.

## External repos:
Informer and DLinear models were trained using their respective repositories
For chronos, one needs autogluon library and it requires Python3.11+