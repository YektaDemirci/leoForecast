#!/usr/bin/env bash

#   ./run_alpha_study.sh                            # the full study, all rates
#   N_REP=2 TEST_HOURS=0.1 ./run_alpha_study.sh     # quick end-to-end check
#   ONLY=exp1 ./run_alpha_study.sh                  # one experiment
#   BEAM_RATES=833e6 ./run_alpha_study.sh           # one rate only
#   DTS_ASYM="0.015 0.15 1.5 15 60" ./run_alpha_study.sh   # 3-5 vs dt too

set -u -o pipefail
cd "$(dirname "$0")"

PY=${PY:-../vv/bin/python}
OUT_ROOT=${OUT_ROOT:-results_alpha}

TEST_HOURS=${TEST_HOURS:-2}
N_REP=${N_REP:-10}

# BEAM_RATES=${BEAM_RATES:-${BEAM_RATE:-"833e6 790e6"}}
#BEAM_RATES=${BEAM_RATES:-${BEAM_RATE:-"790e6"}}
#BEAM_RATES=${BEAM_RATES:-${BEAM_RATE:-"938e6"}}
BEAM_RATES=${BEAM_RATES:-${BEAM_RATE:-"760e6"}}
BUFFER_MB=${BUFFER_MB:-1}
TRAIN_SAMPLES=${TRAIN_SAMPLES:-4200}
PLANNERS=${PLANNERS:-"linearp dlinear"}

DTS_GRAN=${DTS_GRAN:-"0.015 0.15 1.5 15 60"}   # experiment 1
DT_REF=${DT_REF:-"1.5"}                        # experiments 2-5
DTS_ASYM=${DTS_ASYM:-"$DT_REF"}                # widen to sweep 3-5 vs dt
ONLY=${ONLY:-}

# name : sample_dts : alphas : users : fixed-weights
SCENARIOS=(
   #"exp0_alpha_a1p04         : $DT_REF   : 1.04           : 500 500 500 : 1 1 1"

  # 1 -- granularity, everything else held flat
  "exp1_gran_a1p04          : $DTS_GRAN : 1.04           : 500 500 500 : 1 1 1"

  # 2 -- burstiness at the reference granularity
  "exp2_alpha_a1p04         : $DT_REF   : 1.04           : 500 500 500 : 1 1 1"
  "exp2_alpha_a1p24         : $DT_REF   : 1.24           : 500 500 500 : 1 1 1"
  "exp2_alpha_a1p44         : $DT_REF   : 1.44           : 500 500 500 : 1 1 1"

  # 3 -- load asymmetry at constant rho, uniform burstiness
  "exp3_users_900-450-150   : $DTS_ASYM : 1.04           : 900 450 150 : 6 3 1"

  # 4 -- load asymmetry with the biggest cell also the burstiest (realistic)
  "exp4_users_alpha         : $DTS_ASYM : 1.04 1.24 1.44 : 900 450 150 : 6 3 1"
)

mkdir -p "$OUT_ROOT"
t_all=$(date +%s)


rate_tag() { echo "beam${1//[^0-9a-zA-Z]/}"; }

n_rates=$(set -- $BEAM_RATES; echo $#)
echo "beam-hopping study: ${#SCENARIOS[@]} scenarios x ${N_REP} replicates"
echo "                    x ${n_rates} beam rate(s) = arms below"
echo "  policies     ${PLANNERS}"
echo "  test         ${TEST_HOURS} h/replicate"
echo "  beam         ${BEAM_RATES}   buffer ${BUFFER_MB} MiB"
for r in $BEAM_RATES; do
  echo "                 $(rate_tag "$r"): rho = $(awk -v r="$r" \
      'BEGIN{printf "%.3f", 750e6/r}')   -> ${OUT_ROOT}/$(rate_tag "$r")/"
done
echo "  train        ${TRAIN_SAMPLES} samples/granularity, refit per replicate"
echo "  out          ${OUT_ROOT}/"
[ -n "$ONLY" ] && echo "  filter       ONLY=${ONLY}"
echo

for BEAM_RATE in $BEAM_RATES; do
rtag=$(rate_tag "$BEAM_RATE")
echo "##############################################################"
echo "## BEAM RATE ${BEAM_RATE}   (${rtag})"
echo "##############################################################"

for entry in "${SCENARIOS[@]}"; do
  IFS=':' read -r name dts alphas users weights <<< "$entry"
  # strip the padding used to keep the table above readable
  name=$(echo "$name" | xargs); dts=$(echo "$dts" | xargs)
  alphas=$(echo "$alphas" | xargs); users=$(echo "$users" | xargs)
  weights=$(echo "$weights" | xargs)

  [ -n "$ONLY" ] && case "$name" in *"$ONLY"*) ;; *) continue ;; esac

  dir="$OUT_ROOT/$rtag/$name"
  if [ -f "$dir/summary_pooled.txt" ]; then
    echo "== $name: already done ($dir), skipping"
    continue
  fi

  echo "=============================================================="
  echo "== $name   [$rtag]"
  echo "==   sample_dts $dts"
  echo "==   alpha $alphas   users $users   fixed split $weights"
  echo "=============================================================="
  mkdir -p "$dir"
  t0=$(date +%s)

  # shellcheck disable=SC2086  # word splitting is intended for the list args
  "$PY" granularity_study.py \
      --sample-dts ${dts} \
      --test-hours "${TEST_HOURS}" \
      --n-replicates "${N_REP}" \
      --alpha ${alphas} \
      --n-sources ${users} \
      --fixed-weights ${weights} \
      --planners ${PLANNERS} \
      --beam-rate "${BEAM_RATE}" \
      --buffer-mb "${BUFFER_MB}" \
      --train-samples "${TRAIN_SAMPLES}" \
      --out-dir "$dir" 2>&1 | tee "$dir/run.log"

  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "!! $name FAILED (exit $rc) -- see $dir/run.log" >&2
    echo "!! continuing with the remaining scenarios" >&2
    continue
  fi
  echo "== $name done in $(( $(date +%s) - t0 ))s"
  echo
done
done   # BEAM_RATE

echo "=============================================================="
echo "study finished in $(( $(date +%s) - t_all ))s"
echo


echo "linearP H on the grid boundary (clamped, not fitted):"
edge=0
for rate in $BEAM_RATES; do
  rtag=$(rate_tag "$rate")
  for entry in "${SCENARIOS[@]}"; do
    name=$(echo "${entry%%:*}" | xargs)
    f="$OUT_ROOT/$rtag/$name/run.log"
    [ -f "$f" ] || continue
    n=$(grep -c "GRID EDGE" "$f" 2>/dev/null || true)
    if [ "${n:-0}" -gt 0 ]; then
      echo "  $rtag/$name: $n fit(s) pinned -- grep 'GRID EDGE' $f"
      edge=1
    fi
  done
done
[ "$edge" -eq 0 ] && echo "  none -- every linearP H landed inside 0.50 < H < 0.995"
echo

# Grouped by rate, not interleaved: the two arms sit at different rho, so the
# numbers are read WITHIN an arm and only the ranking is carried across.
echo "paired policy comparisons (negative = first policy better):"
for rate in $BEAM_RATES; do
  rtag=$(rate_tag "$rate")
  echo "=== ${rate}  rho = $(awk -v r="$rate" 'BEGIN{printf "%.3f", 750e6/r}')"
  for entry in "${SCENARIOS[@]}"; do
    name=$(echo "${entry%%:*}" | xargs)
    f="$OUT_ROOT/$rtag/$name/summary_pooled.txt"
    [ -f "$f" ] || { echo "  $name: missing"; continue; }
    echo "--- $name"
    grep -E "^sample_dt|^ +\*? *dmax\[" "$f" | sed 's/^/  /'
  done
done
