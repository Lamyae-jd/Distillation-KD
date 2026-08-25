#!/usr/bin/env bash
# Regenerate the HLS4ML Verilog RTL for the StudentNet.
# Idempotent — skips if RTL already present at RTL_SRC_DIR.
#
# Because Vitis HLS refuses paths that contain spaces, the project is
# always driven from /tmp/hls_work (a plain directory populated from the
# in-tree hls_output_rf16). env.sh points HLS_PROJECT_DIR there.
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/env.sh"

if compgen -G "$RTL_SRC_DIR/*.v" > /dev/null; then
  echo "[gen_rtl] RTL already present in $RTL_SRC_DIR — skipping Vitis HLS."
  exit 0
fi

if ! command -v vitis_hls >/dev/null 2>&1; then
  echo "[gen_rtl] ERROR: vitis_hls not in PATH (env.sh should have set it)" >&2
  exit 1
fi

# Seed /tmp/hls_work from the in-tree project if it isn't already populated.
IN_TREE="$( dirname "$_ASIC_ROOT_REAL" )/hls_output_rf16"
if [ "$HLS_PROJECT_DIR" = "/tmp/hls_work" ] && [ ! -f "$HLS_PROJECT_DIR/build_prj.tcl" ]; then
    echo "[gen_rtl] Seeding /tmp/hls_work from $IN_TREE ..."
    mkdir -p "$HLS_PROJECT_DIR"
    cp -a "$IN_TREE"/*.tcl "$IN_TREE"/*.cpp "$IN_TREE"/*.yml "$IN_TREE"/*.onnx "$HLS_PROJECT_DIR/"
    cp -a "$IN_TREE/firmware" "$IN_TREE/tb_data" "$HLS_PROJECT_DIR/"
fi

echo "[gen_rtl] Vitis HLS: $(vitis_hls -version 2>&1 | head -1)"
echo "[gen_rtl] Running vitis_hls in $HLS_PROJECT_DIR (synth only) ..."
cd "$HLS_PROJECT_DIR"
vitis_hls -f build_prj.tcl "csim=0 synth=1 cosim=0 validation=0 export=0" \
    2>&1 | tee "$ASIC_ROOT/work/vitis_hls_gen.log"

if ! compgen -G "$RTL_SRC_DIR/*.v" > /dev/null; then
  echo "[gen_rtl] ERROR: no *.v files produced under $RTL_SRC_DIR" >&2
  exit 2
fi

echo "[gen_rtl] OK — $(ls "$RTL_SRC_DIR"/*.v | wc -l) Verilog files generated."
