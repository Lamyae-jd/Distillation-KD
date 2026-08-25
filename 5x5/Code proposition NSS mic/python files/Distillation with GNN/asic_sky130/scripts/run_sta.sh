#!/usr/bin/env bash
# OpenSTA runner for the post-Yosys synth_netlist.v.
# Reads the gate-level netlist produced by run_synth.sh and runs sta.tcl
# with the env vars sta.tcl expects (CLK_PORT, PROBE_PERIOD_NS).
#
# Usage:
#   bash scripts/run_sta.sh                    # defaults: ap_clk, 1.0 ns
#   CLK_PORT=clk PROBE_PERIOD_NS=2.0 bash scripts/run_sta.sh
#
# Output: work/sta_report.txt (full sta.tcl stdout).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

# env.sh doesn't set WORK_DIR (only run_synth.sh does) — set it here so
# both this script and sta.tcl see the same location as run_synth.sh.
export WORK_DIR="$ASIC_ROOT/work"

# STA-specific env vars (sta.tcl reads these via $::env(...)).
export CLK_PORT="${CLK_PORT:-ap_clk}"
export PROBE_PERIOD_NS="${PROBE_PERIOD_NS:-1.0}"

STA_BIN="$ASIC_ROOT/tools/OpenSTA/build/sta"
NETLIST="$WORK_DIR/synth_netlist.v"

# Guard: STA binary must exist.
if [ ! -x "$STA_BIN" ]; then
    echo "[run_sta] ERROR: OpenSTA binary not found at $STA_BIN" >&2
    echo "[run_sta] Rebuild with: cd tools/OpenSTA/build && make -j" >&2
    exit 1
fi

# Guard: netlist must exist (run_synth.sh must have completed first).
if [ ! -f "$NETLIST" ]; then
    echo "[run_sta] ERROR: gate-level netlist not found at $NETLIST" >&2
    echo "[run_sta] Run 'bash scripts/run_synth.sh --skip-hls' first." >&2
    exit 2
fi

# WORK_DIR is not exported by env.sh (only run_synth.sh sets it); set it
# here so sta.tcl finds synth_netlist.v in the same directory as the yosys
# output.
export WORK_DIR

STA_LOG="$WORK_DIR/sta_report.txt"
echo "[run_sta] OpenSTA : $($STA_BIN -version 2>&1 | head -1)"
echo "[run_sta] Netlist : $NETLIST"
echo "[run_sta] Clock   : $CLK_PORT @ ${PROBE_PERIOD_NS} ns probe"
echo "[run_sta] Log     : $STA_LOG"

"$STA_BIN" -no_splash -exit "$SCRIPT_DIR/sta.tcl" 2>&1 | tee "$STA_LOG"

echo
echo "[run_sta] Done. Summary:"
grep -E "^(PROBE_PERIOD_NS|WORST_SLACK_NS|CRITICAL_DELAY_NS|FMAX_MHZ|N_REGISTERS)" "$STA_LOG" || true
