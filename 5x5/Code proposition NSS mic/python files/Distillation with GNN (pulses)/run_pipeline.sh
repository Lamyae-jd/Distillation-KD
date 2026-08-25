#!/usr/bin/env bash
# Full pulse-teacher pipeline: preprocess -> train -> eval + figures.
# Launched with nohup so it survives disconnect.

set -u
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="logs/pipeline_${STAMP}.log"
PY=/usr/bin/python3

echo "=== [$STAMP] Starting pipeline in $SCRIPT_DIR" | tee -a "$LOG"
echo "=== PID $$ | Python $($PY --version 2>&1)"    | tee -a "$LOG"

step () {
    local name="$1"; shift
    echo "" | tee -a "$LOG"
    echo "==================== STEP: $name ====================" | tee -a "$LOG"
    date -u +"[%Y-%m-%d %H:%M:%SZ] START $name" | tee -a "$LOG"
    ("$@") >>"$LOG" 2>&1
    local rc=$?
    date -u +"[%Y-%m-%d %H:%M:%SZ] END   $name (rc=$rc)" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then
        echo "FAILED at step: $name (rc=$rc). See $LOG" | tee -a "$LOG"
        exit $rc
    fi
}

if [ -f data_cache.npz ] && [ -f data_cache_lists.pkl ]; then
    echo "Cache already present — skipping preprocess." | tee -a "$LOG"
else
    step "preprocess" $PY -u preprocess.py
fi

step "train"           $PY -u train.py
step "eval + figures"  $PY -u eval_and_figures.py

echo "" | tee -a "$LOG"
echo "=== [$(date +%Y%m%d_%H%M%S)] PIPELINE DONE — see $LOG" | tee -a "$LOG"
