#!/bin/bash
# Monitor student training: check after 1h, restart if KD is hurting
# Usage: bash monitor_student.sh &

WORKDIR="/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN"
LOG="$WORKDIR/training_student_log.txt"
REPORT="$WORKDIR/monitor_report.txt"
CONFIG="$WORKDIR/pi_kd/configs/train_config.yaml"
WAIT_SECONDS=3600  # 1 hour

echo "[monitor] Started at $(date). Will check in ${WAIT_SECONDS}s." | tee "$REPORT"
sleep $WAIT_SECONDS

echo "[monitor] Woke up at $(date)" | tee -a "$REPORT"

# ── 1. Is the training still running? ────────────────────────────────────────
TRAIN_PID=$(pgrep -f "pi_kd.scripts.train" | head -1)
if [ -z "$TRAIN_PID" ]; then
    echo "[monitor] ERROR: training process not found. Check log manually." | tee -a "$REPORT"
    tail -50 "$LOG" >> "$REPORT"
    exit 1
fi
echo "[monitor] Training PID=$TRAIN_PID is alive." | tee -a "$REPORT"

# ── 2. Extract last epoch metrics ────────────────────────────────────────────
LAST_LINE=$(grep "Ep " "$LOG" | tail -1)
echo "[monitor] Last epoch line: $LAST_LINE" | tee -a "$REPORT"

# Extract key values
EPOCH=$(echo "$LAST_LINE" | grep -oP 'Ep\s+\K[0-9]+')
SCORE=$(echo "$LAST_LINE" | grep -oP 'score=\K[0-9.]+')
KD=$(echo "$LAST_LINE" | grep -oP 'kd=\K[0-9.]+')
TASK=$(echo "$LAST_LINE" | grep -oP 'task=\K[0-9.]+')
PHASE=$(echo "$LAST_LINE" | grep -oP 'Phase\K[^]]+' | head -1)

echo "[monitor] Epoch=$EPOCH | Phase=$PHASE | score=$SCORE | kd=$KD | task=$TASK" | tee -a "$REPORT"

# ── 3. Check if KD is hurting ────────────────────────────────────────────────
# KD hurts if kd >> task (ratio > 5) during phase 2+
KD_RATIO=$(python3 -c "
kd = float('${KD}') if '${KD}' else 0
task = float('${TASK}') if '${TASK}' else 0.0001
ratio = kd / max(task, 1e-9)
print(f'{ratio:.1f}')
" 2>/dev/null || echo "0")

echo "[monitor] KD/task ratio = $KD_RATIO" | tee -a "$REPORT"

# Check if score dropped below 0.15 (was 0.204 at epoch 12)
SCORE_OK=$(python3 -c "print('yes' if float('${SCORE:-0}') >= 0.15 else 'no')" 2>/dev/null || echo "no")
RATIO_OK=$(python3 -c "print('yes' if float('${KD_RATIO:-0}') < 10 else 'no')" 2>/dev/null || echo "no")

if [ "$SCORE_OK" = "no" ] || [ "$RATIO_OK" = "no" ]; then
    echo "[monitor] PROBLEM DETECTED: score=$SCORE ratio=$KD_RATIO → killing and restarting with lower lambda_kd" | tee -a "$REPORT"

    # Kill training
    pkill -f "pi_kd.scripts.train"
    sleep 3

    # Halve lambda_kd in config
    python3 - <<'PYEOF'
import yaml, re

config_path = "/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/pi_kd/configs/train_config.yaml"
with open(config_path) as f:
    content = f.read()

# Extract current lambda_kd and halve it
match = re.search(r'lambda_kd:\s*([0-9.e+-]+)', content)
if match:
    old_val = float(match.group(1))
    new_val = old_val / 2
    content = re.sub(r'(lambda_kd:\s*)[0-9.e+-]+', f'\\g<1>{new_val:.6f}', content)
    with open(config_path, 'w') as f:
        f.write(content)
    print(f"lambda_kd: {old_val} → {new_val}")
else:
    print("Could not find lambda_kd in config")
PYEOF

    # Restart
    cd "$WORKDIR"
    nohup python3 -m pi_kd.scripts.train >> "$WORKDIR/training_student_log.txt" 2>&1 &
    NEW_PID=$!
    echo "[monitor] Restarted with PID=$NEW_PID" | tee -a "$REPORT"
else
    echo "[monitor] Training looks healthy. score=$SCORE ratio=$KD_RATIO — no action needed." | tee -a "$REPORT"
fi

echo "[monitor] Done at $(date)" | tee -a "$REPORT"
