#!/usr/bin/env bash
# End-to-end ASIC flow: HLS RTL → Yosys → OpenSTA → runs/<date>_<label>/
#
# Usage:
#   bash scripts/run_full.sh --label frac4
#   bash scripts/run_full.sh --label frac4 --skip-hls   # RTL already generated
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ASIC_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

LABEL=""
SKIP_HLS=""
while [ $# -gt 0 ]; do
    case "$1" in
        --label) LABEL="$2"; shift 2 ;;
        --skip-hls) SKIP_HLS="--skip-hls"; shift ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$LABEL" ]; then
    echo "ERROR: --label <name> is required (e.g. --label frac4)" >&2
    exit 2
fi

DATE="$(date +%Y-%m-%d)"
RUN_DIR="$ASIC_ROOT/runs/${DATE}_${LABEL}"

if [ -e "$RUN_DIR" ]; then
    echo "ERROR: run directory already exists: $RUN_DIR" >&2
    echo "       pick a different --label or remove it first" >&2
    exit 3
fi

echo "[run_full] target run: $RUN_DIR"

bash "$SCRIPT_DIR/run_synth.sh" $SKIP_HLS

WORK_DIR="$ASIC_ROOT/work"
mkdir -p "$RUN_DIR"
for artifact in report.txt cell_breakdown.txt opensta.log; do
    src="$WORK_DIR/$artifact"
    [ -s "$src" ] && cp "$src" "$RUN_DIR/$artifact"
done

/usr/bin/python3 "$SCRIPT_DIR/collect_results.py" "$RUN_DIR"

echo
echo "[run_full] done → $RUN_DIR"
echo "[run_full] metrics:"
sed 's/^/    /' "$RUN_DIR/metrics.json" | head -14
