#!/usr/bin/env python3
"""Extract ASIC synthesis metrics from run directories.

Usage:
    python collect_results.py <run_dir>   # process one run + refresh summary
    python collect_results.py --all       # (re)process every runs/*/

Writes:
    <run_dir>/metrics.json    # all metrics for that run
    runs/summary.csv          # one row per run
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Optional

RUNS_ROOT = Path(__file__).resolve().parent.parent / "runs"
SUMMARY_CSV = RUNS_ROOT / "summary.csv"

DFF_CELL_TYPES = ("dfxtp_1", "edfxtp_1", "dfrtp_1", "dfstp_1")

SUMMARY_FIELDS = [
    "label", "date", "top_module",
    "total_cells", "top_module_area_um2", "kge",
    "total_dffs", "edfxtp_1", "dfxtp_1",
    "critical_path_ps", "fmax_abc_mhz",
    "sta_worst_slack_ns", "sta_fmax_mhz", "sta_n_registers",
]


def _search(pattern: str, text: str, flags: int = 0) -> Optional[str]:
    m = re.search(pattern, text, flags)
    return m.group(1) if m else None


def _to_float(v: Optional[str]) -> Optional[float]:
    return float(v) if v is not None else None


def _to_int(v: Optional[str]) -> Optional[int]:
    return int(v) if v is not None else None


def _top_module_area_um2(text: str, top_module: Optional[str]) -> Optional[float]:
    if not top_module:
        return None
    # Yosys prints `\myproject` (escaped) for the top module in its area dump.
    pat = rf"Chip area for module '\\{re.escape(top_module)}':\s+([\d.]+)"
    return _to_float(_search(pat, text))


def parse_report(path: Path) -> dict:
    text = path.read_text()
    top_module = _search(r"Top\s*:\s*(\S+)", text)
    return {
        "date": _search(r"Date\s*:\s*(\S+)", text),
        "top_module": top_module,
        "top_module_area_um2": _top_module_area_um2(text, top_module),
        "critical_path_ps": _to_float(_search(r"Critical-path delay\s*:\s*([\d.]+)\s*ps", text)),
        "fmax_abc_mhz": _to_float(_search(r"Fmax\s*\(1/delay\)\s*:\s*([\d.]+)\s*MHz", text)),
        "kge": _to_float(_search(r"Gate equivalent\s*:\s*([\d.]+)\s*kGE", text)),
    }


def parse_cell_breakdown(path: Path) -> dict:
    """Parse cell_breakdown.txt (`<count> <area> sky130_fd_sc_hd__<cell>` lines)."""
    text = path.read_text()
    total_cells = _to_int(_search(r"TOTAL:\s+(\d+)", text))
    counts: dict[str, int] = {}
    for m in re.finditer(r"^\s*(\d+)\s+\S+\s+sky130_fd_sc_hd__(\w+)\s*$", text, re.M):
        counts[m.group(2)] = int(m.group(1))
    total_dffs = sum(counts.get(name, 0) for name in DFF_CELL_TYPES) or None
    return {
        "total_cells": total_cells,
        "cell_counts": counts,
        "total_dffs": total_dffs,
        "edfxtp_1": counts.get("edfxtp_1"),
        "dfxtp_1": counts.get("dfxtp_1"),
    }


def parse_opensta_log(path: Path) -> dict:
    text = path.read_text()
    return {
        "sta_worst_slack_ns": _to_float(_search(r"^WORST_SLACK_NS\s+(\S+)", text, re.M)),
        "sta_fmax_mhz": _to_float(_search(r"^FMAX_MHZ\s+(\S+)", text, re.M)),
        "sta_n_registers": _to_int(_search(r"^N_REGISTERS\s+(\d+)", text, re.M)),
    }


def collect(run_dir: Path) -> dict:
    report = run_dir / "report.txt"
    if not report.exists():
        raise FileNotFoundError(f"report.txt missing in {run_dir}")

    metrics: dict = {"label": run_dir.name}
    metrics.update(parse_report(report))

    breakdown = run_dir / "cell_breakdown.txt"
    if breakdown.exists():
        metrics.update(parse_cell_breakdown(breakdown))

    opensta = run_dir / "opensta.log"
    if opensta.exists() and opensta.stat().st_size > 0:
        metrics.update(parse_opensta_log(opensta))

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def write_summary(rows: list[dict]) -> None:
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def process_all() -> list[dict]:
    rows = []
    for run in sorted(p for p in RUNS_ROOT.iterdir() if p.is_dir()):
        if not (run / "report.txt").exists():
            continue
        rows.append(collect(run))
        print(f"[collect] {run.name}  →  metrics.json")
    write_summary(rows)
    print(f"[collect] summary.csv  →  {len(rows)} runs")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, help="single run directory to process")
    parser.add_argument("--all", action="store_true", help="reprocess all runs/*/")
    args = parser.parse_args()

    if args.all or args.run_dir is None:
        process_all()
    else:
        collect(args.run_dir)
        process_all()


if __name__ == "__main__":
    main()
