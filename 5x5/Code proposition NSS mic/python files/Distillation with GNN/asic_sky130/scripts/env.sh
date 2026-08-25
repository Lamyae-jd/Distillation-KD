#!/usr/bin/env bash
# Environment for the ASIC synthesis flow.
# Source this before running any yosys / opensta command:
#   source scripts/env.sh

# Always resolve to a *physical* path first: readlink -f follows any
# symlinks so we don't accidentally create /tmp/asic_work -> /tmp/asic_work
# when env.sh is sourced from within the alias.
export _ASIC_ROOT_REAL="$( readlink -f "$( dirname "${BASH_SOURCE[0]}" )/.." )"

# ABC (called by yosys) chokes on paths containing spaces when they show up
# in -script / -constr args. Publish a space-free alias in /tmp so every
# path used inside the flow is safe.
_ASIC_ALIAS="/tmp/asic_work"
if [ "$_ASIC_ROOT_REAL" != "$_ASIC_ALIAS" ]; then
  if [ ! -L "$_ASIC_ALIAS" ] || [ "$(readlink "$_ASIC_ALIAS")" != "$_ASIC_ROOT_REAL" ]; then
    ln -sfn "$_ASIC_ROOT_REAL" "$_ASIC_ALIAS"
  fi
fi

export ASIC_ROOT="$_ASIC_ALIAS"
export PDK_LIB="$ASIC_ROOT/pdk/sky130_fd_sc_hd__tt_025C_1v80.lib"
export OSS_CAD_ROOT="$ASIC_ROOT/tools/oss-cad-suite"

if [ -d "$OSS_CAD_ROOT" ]; then
  source "$OSS_CAD_ROOT/environment"
fi

# settings64.sh in this install references /opt/Xilinx paths that don't
# exist here — the tools actually live under /data/.Xilinx. Publish the
# minimal env vars needed to run vitis_hls directly.
export XILINX_HLS="/data/.Xilinx/Vitis_HLS/2023.1"
export PATH="$XILINX_HLS/bin:$PATH"
export LD_LIBRARY_PATH="$XILINX_HLS/lnx64/tools/fpo_v7_1:$XILINX_HLS/lnx64/lib/csim:$XILINX_HLS/lnx64/tools/fft_v9_1:$XILINX_HLS/lnx64/tools/fir_v7_0:$XILINX_HLS/lnx64/tools/dds_v6_0:${LD_LIBRARY_PATH:-}"

# HLS project — Vitis HLS refuses paths with spaces, so it always runs
# from a disk-backed dir with no spaces. Preferred: /data/jdil1901/hls_work
# (disk, safer than tmpfs for multi-GB Vitis runs). Fallbacks : /tmp/hls_work
# (older runs), then hls_output_rf16/ in-tree (never populated as-is).
_HLS_DISK="/data/jdil1901/hls_work"
_HLS_TMP="/tmp/hls_work"
_HLS_INTREE="$( dirname "$_ASIC_ROOT_REAL" )/hls_output_rf16"
if [ -d "$_HLS_DISK/myproject_prj/solution1/syn/verilog" ]; then
    export HLS_PROJECT_DIR="$_HLS_DISK"
elif [ -d "$_HLS_TMP/myproject_prj/solution1/syn/verilog" ]; then
    export HLS_PROJECT_DIR="$_HLS_TMP"
elif [ -d "$_HLS_INTREE/myproject_prj/solution1/syn/verilog" ]; then
    export HLS_PROJECT_DIR="$_HLS_INTREE"
else
    # Nothing populated yet; default to the disk-backed location so gen_rtl.sh
    # writes there (matches /tmp/asic_bg_run.sh).
    export HLS_PROJECT_DIR="$_HLS_DISK"
fi
export RTL_SRC_DIR="${RTL_SRC_DIR:-$HLS_PROJECT_DIR/myproject_prj/solution1/syn/verilog}"
export TOP_MODULE="${TOP_MODULE:-myproject}"

# OpenSTA inputs. Vitis HLS names the clock ap_clk. PROBE_PERIOD_NS is only a
# measuring stick: Fmax = 1/(probe - slack) holds whether slack is +/-.
export STA_BIN="$ASIC_ROOT/tools/OpenSTA/build/sta"
export CLK_PORT="${CLK_PORT:-ap_clk}"
export PROBE_PERIOD_NS="${PROBE_PERIOD_NS:-1.0}"

echo "[env] ASIC_ROOT=$ASIC_ROOT"
echo "[env] PDK_LIB=$PDK_LIB"
echo "[env] RTL_SRC_DIR=$RTL_SRC_DIR"
echo "[env] TOP_MODULE=$TOP_MODULE"
