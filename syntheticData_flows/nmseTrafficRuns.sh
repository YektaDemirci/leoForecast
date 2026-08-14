#!/bin/bash
# python venv: vv (has torch + the R/rpy2 stack farima_f.py needs).
# Runtime: the ARFIMA MLE on ~1000 samples is fast -- a few minutes total.
set -euo pipefail
cd "$(dirname "$0")"

PY=../vv/bin/python
export PYTHONPATH=..

DATA_DIR=./nmse_traffic
DT=1
# LEN=1440*5                # trace length [s] -> LEN/DT samples
LEN=7200
T=48                    # predictor taps
HORIZON=1
P=1                     # same ARFIMA(p,d,0) AR order wifiRuns.sh settled on
OUT_DIR=results
mkdir -p "$OUT_DIR"

if [ ! -f "$DATA_DIR/manifest.csv" ]; then
    echo "no traces in $DATA_DIR -- generating"
    $PY gen_nmse_traffic.py --dt "$DT" --T "$LEN" --outdir "$DATA_DIR"
fi

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') -- score_nmse_traffic.py H=fit --dt $DT --T $T --horizon $HORIZON --p $P $* ==="
    $PY score_nmse_traffic.py \
        --data-dir "$DATA_DIR" --dt "$DT" \
        --T "$T" --horizon "$HORIZON" --p "$P" \
        --per-fc-out "$OUT_DIR/fc_nmse_traffic_fit" \
        --out "$OUT_DIR/mase_traffic.csv" \
        --res-txt "$OUT_DIR/mase_traffic_res.txt" \
        --summary-out "$OUT_DIR/mase_traffic_summary.txt" "$@"
    echo
} | tee -a "$OUT_DIR/mase_traffic.txt"
