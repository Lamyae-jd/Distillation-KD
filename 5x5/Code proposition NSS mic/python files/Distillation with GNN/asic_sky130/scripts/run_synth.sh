#!/usr/bin/env bash
# Master flow: HLS RTL -> Yosys synth on SKY130 -> parse -> report.txt
#
# Usage:
#   bash scripts/run_synth.sh              # full flow (gen RTL + synth + report)
#   bash scripts/run_synth.sh --smoke      # use built-in smoke.v (for pipeline validation)
#   bash scripts/run_synth.sh --skip-hls   # assume RTL already present, skip Vitis HLS
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

MODE="full"
for arg in "$@"; do
  case "$arg" in
    --smoke)    MODE="smoke" ;;
    --skip-hls) MODE="skip-hls" ;;
    *) echo "unknown flag: $arg" >&2; exit 2 ;;
  esac
done

WORK_DIR="$ASIC_ROOT/work"
export WORK_DIR
mkdir -p "$WORK_DIR"

case "$MODE" in
  smoke)
    export RTL_SRC_DIR="$WORK_DIR/smoke_src"
    export TOP_MODULE="smoke"
    echo "[run_synth] MODE=smoke — using $RTL_SRC_DIR/*.v"
    ;;
  skip-hls)
    echo "[run_synth] MODE=skip-hls — expecting Verilog already in $RTL_SRC_DIR"
    ;;
  full)
    bash "$SCRIPT_DIR/gen_rtl.sh"
    ;;
esac

SYNTH_LOG="$WORK_DIR/yosys_synth.log"
echo "[run_synth] Yosys synth (log: $SYNTH_LOG) ..."
yosys -Q -l "$SYNTH_LOG" -c "$SCRIPT_DIR/synth.ys"

STA_LOG="$WORK_DIR/opensta.log"
if [ "$MODE" != "smoke" ] && [ -x "$STA_BIN" ]; then
  echo "[run_synth] OpenSTA (clk=$CLK_PORT, probe=${PROBE_PERIOD_NS}ns, log: $STA_LOG) ..."
  # Pass sta.tcl via the space-free ASIC_ROOT alias — OpenSTA's -exit re-tokenizes
  # its argument internally, so a path with spaces gets split and sta::include_file
  # fails with "wrong # args: should be sta::include_file filename echo verbose".
  "$STA_BIN" -no_splash -exit "$ASIC_ROOT/scripts/sta.tcl" > "$STA_LOG" 2>&1 \
    || echo "[run_synth] WARNING: OpenSTA exited non-zero — see $STA_LOG"
else
  echo "[run_synth] Skipping OpenSTA (smoke mode or $STA_BIN missing)."
fi

REPORT="$WORK_DIR/report.txt"
{
  echo "======================================================================"
  echo "  StudentNet ASIC synthesis on SKY130 (sky130_fd_sc_hd)"
  echo "  Corner: typical (tt_025C_1v80)"
  echo "  Date  : $(date -Iseconds)"
  echo "  Top   : $TOP_MODULE"
  echo "  RTL   : $RTL_SRC_DIR"
  echo "======================================================================"
  echo

  echo "----- AREA -----"
  awk '/Chip area for module/{print}' "$SYNTH_LOG"
  echo

  echo "----- CELL COUNTS (post tech-map, top-5 by cumulative area) -----"
  # In stat -liberty output the last "<total> <area> cells" line marks the
  # start of a per-cell-type breakdown: "<count> <cumulative-area> <name>".
  awk '
    /^[[:space:]]+[0-9]+[[:space:]]+[0-9eE.+]+[[:space:]]+cells$/ {
        block=""; capture=1; total_line=$0; next
    }
    capture && /^[[:space:]]+[0-9]+[[:space:]]+[0-9eE.+]+[[:space:]]+sky130_/ {
        block=block $0 "\n"; next
    }
    capture {capture=0; last=block; last_total=total_line}
    END {printf "TOTAL: %s\n%s", last_total, last}
  ' "$SYNTH_LOG" > "$WORK_DIR/cell_breakdown.txt"
  # `sort | head -5` triggers SIGPIPE on `sort` when `head` closes early; with
  # `set -o pipefail` at top of script that becomes exit 141 and kills the
  # whole `{ ... } > "$REPORT"` group. `|| true` swallows the SIGPIPE.
  head -1 "$WORK_DIR/cell_breakdown.txt" || true
  tail -n +2 "$WORK_DIR/cell_breakdown.txt" | sort -k2 -g -r | head -5 || true
  echo

  echo "----- TIMING (ABC critical-path, WireLoad=none corner) -----"
  DELAY_PS=$(awk -F 'Delay = *' '/WireLoad = "none".*Delay =/ {gsub(/ ps.*/,"",$2); v=$2} END{print v}' "$SYNTH_LOG")
  if [ -n "$DELAY_PS" ]; then
    FMAX_MHZ=$(awk -v d="$DELAY_PS" 'BEGIN{ if (d>0) printf "%.1f\n", 1e6/d; else print "n/a" }')
    printf "Critical-path delay : %s ps\n" "$DELAY_PS"
    printf "Fmax (1/delay)      : %s MHz\n" "$FMAX_MHZ"
  else
    echo "(no ABC delay summary found in log)"
  fi
  echo

  NAND2_AREA=$(awk '/cell \("sky130_fd_sc_hd__nand2_1"\)/{f=1} f && /area/{print $NF; exit}' "$PDK_LIB" | tr -d ';')
  CHIP_AREA=$(awk '/Chip area for module.*:/{print $NF; exit}' "$SYNTH_LOG")
  if [ -n "$NAND2_AREA" ] && [ -n "$CHIP_AREA" ]; then
    KGE=$(awk -v c="$CHIP_AREA" -v n="$NAND2_AREA" 'BEGIN{ if (n>0) printf "%.2f\n", c/n/1000.0; else print "n/a" }')
    echo "----- GATE EQUIVALENT -----"
    printf "NAND2 reference area : %s um^2\n" "$NAND2_AREA"
    printf "Chip area            : %s um^2\n" "$CHIP_AREA"
    printf "Gate equivalent      : %s kGE\n" "$KGE"
    echo
  fi

  if [ -f "$STA_LOG" ]; then
    echo "----- TIMING (OpenSTA, sign-off style, tt_025C_1v80) -----"
    awk '/^PROBE_PERIOD_NS|^WORST_SLACK_NS|^CRITICAL_DELAY_NS|^FMAX_MHZ|^N_REGISTERS/{print}' "$STA_LOG"
    echo
    echo "----- POWER (OpenSTA, default activity 10%/50% — order of magnitude only) -----"
    awk '/^Total  /{print "Total power (W)      :", $5}' "$STA_LOG"
    echo
  fi

  echo "----- DISCLAIMER -----"
  echo "Synthesis only, no place-and-route. Area under-estimates final by"
  echo "~30% (no routing tracks); Fmax over-estimates by ~30% (no long-wire"
  echo "delays). Corner: typical (tt_025C_1v80) only."
} > "$REPORT"

echo "[run_synth] Done. Report at: $REPORT"
echo
cat "$REPORT"
