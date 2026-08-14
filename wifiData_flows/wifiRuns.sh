#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

PY=../vv/bin/python
export PYTHONPATH=..

T=48
HORIZON=1

OUT_DIR=results
# Named for the metric they carry: the columns are mase_*, so the old
# wifi_nmse.* files are left alone rather than appended to under a schema they
# do not share (score_wifi.py refuses a column mismatch anyway).
OUT_CSV="$OUT_DIR/wifi_mase.csv"
OUT_TXT="$OUT_DIR/wifi_mase.txt"
mkdir -p "$OUT_DIR"

# Per-forecast dumps are rebuilt below. Clearing them first stops a stale dump
# from an earlier dataset being read as if it belonged to this run.
rm -f "$OUT_DIR"/fc_*.csv

# The trace now lives beside this script; the root baselines reach in via
# wifiData_flows/.
DATA=wifiData.csv

P_VALUES="1"

for P in $P_VALUES; do
{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') -- score_wifi.py RAW  --T $T --horizon $HORIZON --p $P $* ==="
    $PY score_wifi.py \
        --csv "$DATA" --col OT --no-deseason \
        --T "$T" --horizon "$HORIZON" --p "$P" \
        --fc-out "$OUT_DIR/fc_analytic_raw_p$P.csv" \
        --out "$OUT_CSV" "$@"
    echo

    echo "=== $(date '+%Y-%m-%d %H:%M:%S') -- score_wifi.py DESEASON  --T $T --horizon $HORIZON --p $P $* ==="
    $PY score_wifi.py \
        --csv "$DATA" --col OT \
        --T "$T" --horizon "$HORIZON" --p "$P" \
        --fc-out "$OUT_DIR/fc_analytic_deseason_p$P.csv" \
        --out "$OUT_CSV" "$@"
    echo

} | tee -a "$OUT_TXT"
done
