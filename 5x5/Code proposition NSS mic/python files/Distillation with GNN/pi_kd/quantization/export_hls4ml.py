"""
Export QAT-adapted StudentNet to HLS via hls4ml (ONNX backend).

Pipeline:
  best_qat.pt  -> load_for_hls4ml()  (BN-fused float32 weights)
               -> HLS4MLWrapper      (single-tensor output)
               -> ONNX opset 11      (PyTorch export)
               -> qonnx channels-last (NHWC for hls4ml)
               -> sanitize ONNX names (digit-leading names crash Vitis HLS)
               -> hls4ml ONNX converter
               -> HLS C++
               -> Vitis HLS synthesis (csynth only, ~10-30 min with rf=16)
               -> csynth.rpt -> latency / II / resources / throughput

WHY ONNX BACKEND (not PyTorch):
  hls4ml's PyTorch Conv2d handler (converters/pytorch/convolution.py) reads
  out_channels, kernel_size, stride — but NEVER reads class_object.groups.
  Depthwise convolutions (groups=in_channels) are silently mapped to regular
  Conv2D: wrong weight shape, wrong number of multiplications.
  The ONNX backend reads the 'group' attribute from ONNX Conv nodes and
  correctly emits nnet::depthwise_conv_2d_cl. Non-negotiable for StudentNet.

WHY reuse_factor=16:
  With rf=1 (fully unrolled), DATAFLOW FIFOs alone cost 850K LUT.
  Total = 1.78M LUT vs ZCU104's 230K (775% — does not fit).
  rf=16 reduces compute + FIFOs ~proportionally → estimated 112K LUT (49%).
  Tradeoff: II grows from 25 to ~400 cycles; at 200 MHz: 0.5 M inf/s.

WHY DSP=0 (and it's correct):
  ap_fixed<8,4> = 8-bit multiplication. The DSP48E2 on UltraScale+ is a
  27x18-bit multiplier. Using it for 8x8 bits wastes 87% of its capacity;
  a LUT-based 8-bit multiplier costs 4-6 LUTs and is more area-efficient.
  Vitis HLS is correct to choose LUTs. DSPs are for float16/32 or >=18-bit
  fixed-point. To force DSPs: use ap_fixed<18,8> precision (wider = DSP
  threshold), but that requires 179K DSPs for rf=1 — physically impossible.
  With rf=128+, a few hundred DSPs become feasible (future work).
"""

import copy
import os
import re
import shutil
import sys
import torch
import torch.nn as nn

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

VITIS_HLS_BIN = "/data/.Xilinx/Vitis_HLS/2023.1/bin"


# ─────────────────────────────────────────────────────────────────
# Model preparation
# ─────────────────────────────────────────────────────────────────

def replace_relu6(model):
    """Return deep copy of model with every ReLU6 replaced by ReLU."""
    m = copy.deepcopy(model)
    for name, module in m.named_modules():
        if isinstance(module, nn.ReLU6):
            parts = name.split(".")
            parent = m
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], nn.ReLU(inplace=module.inplace))
    return m


def load_for_hls4ml(qat_ckpt_path, device="cpu"):
    """
    Load QAT checkpoint into a clean BN-fused StudentNet.

    The QAT state_dict contains extra fake-quantizer keys (weight_fake_quant.*,
    activation_post_process.*) that hls4ml cannot parse. This function builds a
    fresh fused model and loads only the matching keys.
    """
    from ..models.student import StudentNet

    ckpt = torch.load(qat_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    widths = tuple(cfg.get("student", {}).get("widths", [32, 64, 64]))

    model = StudentNet(in_ch=3, widths=widths)
    model.eval()
    model.fuse_bn()

    fused_keys = set(model.state_dict().keys())
    qat_sd = ckpt["student"]
    filtered = {k: v for k, v in qat_sd.items() if k in fused_keys}

    missing = fused_keys - set(filtered.keys())
    if missing:
        print(f"  WARNING: {len(missing)} keys not found in QAT checkpoint: {missing}")

    model.load_state_dict(filtered, strict=False)
    model.eval()

    metrics = ckpt.get("metrics", {})
    print(f"Loaded QAT checkpoint (epoch {ckpt.get('epoch', '?')})")
    print(f"  AccTop1={metrics.get('AccTop1', 0):.4f}  "
          f"AUPRC_Primary_ICS={metrics.get('AUPRC_Primary_ICS', 0):.4f}")
    print(f"  Model: StudentNet (BN fused), {model.count_params()} params")
    return model, cfg


class HLS4MLWrapper(nn.Module):
    """
    Wraps StudentNet for hls4ml: single tensor output, no dynamic control flow.
    Exposes backbone + primary_head only (P_logits: B×1×5×5).
    """

    def __init__(self, student):
        super().__init__()
        self.backbone = student.backbone
        self.primary_head = student.primary_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.primary_head(self.backbone(x))


# ─────────────────────────────────────────────────────────────────
# Profiling: activation range check before synthesis
# ─────────────────────────────────────────────────────────────────

def profile_activations(wrapper, n_samples=512, precision_int_bits=4, total_bits=8):
    """
    Forward n_samples random inputs through PyTorch wrapper and report
    per-layer activation statistics. Flags layers that may clip in ap_fixed.

    ap_fixed<8,4>: 1 sign + 3 integer + 4 fractional → range [-8, 7.875].
    Values beyond this range will be saturated by hls4ml.
    """
    max_val = 2 ** (precision_int_bits - 1) - 2 ** (-(total_bits - precision_int_bits))
    print(f"\n── Activation profiling  (ap_fixed<{total_bits},{precision_int_bits}> "
          f"range = [-{2**(precision_int_bits-1)}, {max_val:.3f}]) ──")

    handles, stats = [], {}

    def make_hook(name):
        def hook(module, inp, out):
            t = out.detach().float()
            stats[name] = {
                "max_abs": t.abs().max().item(),
                "mean":    t.mean().item(),
                "std":     t.std().item(),
            }
        return hook

    for name, module in wrapper.named_modules():
        if isinstance(module, (nn.Conv2d, nn.ReLU, nn.ReLU6)):
            handles.append(module.register_forward_hook(make_hook(name)))

    dummy = torch.randn(n_samples, 3, 5, 5) * 0.5
    with torch.no_grad():
        wrapper(dummy)

    for h in handles:
        h.remove()

    clipped = []
    for name, s in stats.items():
        flag = "  ⚠ CLIP" if s["max_abs"] > max_val else ""
        print(f"  {name:45s}  max_abs={s['max_abs']:6.3f}  "
              f"mean={s['mean']:+.3f}  std={s['std']:.3f}{flag}")
        if s["max_abs"] > max_val:
            clipped.append(name)

    if clipped:
        print(f"\n  WARNING: {len(clipped)} layer(s) may clip: {clipped}")
        print(f"  → Consider ap_fixed<{total_bits},{precision_int_bits+1}> or wider.")
    else:
        print(f"  All activations within ap_fixed<{total_bits},{precision_int_bits}> range. ✓")
    return stats


# ─────────────────────────────────────────────────────────────────
# Export: PyTorch → ONNX → hls4ml HLS C++
# ─────────────────────────────────────────────────────────────────

def export_to_hls(model, output_dir, backend="Vivado", part=None, board=None,
                  clock_period=5.0, io_type="io_parallel", reuse_factor=16,
                  precision="fixed<8,4>"):
    """
    Convert HLS4MLWrapper to HLS C++ using the hls4ml ONNX backend.

    The output_dir MUST NOT contain spaces (Vitis HLS hard requirement).
    Use /tmp/hls_pikd_rf16/ or similar.
    """
    if " " in output_dir:
        raise ValueError(
            f"Output dir contains spaces: '{output_dir}'\n"
            f"Vitis HLS cannot handle spaces in paths. Use e.g. /tmp/hls_pikd_rf16/"
        )

    try:
        import hls4ml
        import onnx
    except ImportError as e:
        raise ImportError(f"Missing: {e}\nRun: pip install hls4ml onnx qonnx")

    os.makedirs(output_dir, exist_ok=True)

    # ── 1. PyTorch → ONNX ───────────────────────────────────────────────────
    onnx_path = os.path.join(output_dir, "student.onnx")
    model.eval()
    dummy = torch.randn(1, 3, 5, 5)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["P_logits"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=11,
    )
    print(f"[1/6] ONNX → {onnx_path}")

    # ── 2. qonnx: cleanup + channels-last (NCHW → NHWC) ────────────────────
    try:
        from qonnx.util.cleanup import cleanup
        from qonnx.util.to_channels_last import to_channels_last
        cl_path = onnx_path.replace(".onnx", "_channels_last.onnx")
        cleanup(onnx_path, out_file=onnx_path)
        to_channels_last(onnx_path, out_file=cl_path)
        print(f"[2/6] Channels-last ONNX → {cl_path}")
    except Exception as e:
        print(f"[2/6] WARNING: qonnx failed ({e}), continuing with original ONNX")
        cl_path = onnx_path

    onnx_model = onnx.load(cl_path)
    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    # ── 3. Strip input/output Transpose nodes ───────────────────────────────
    # qonnx inserts Transpose at input (NCHW→NHWC) and output (NHWC→NCHW).
    # hls4ml optimizer crashes on these (empty inputs list). Strip them and
    # fix the graph input shape to NHWC directly.
    nodes = list(onnx_model.graph.node)
    while nodes and nodes[0].op_type == "Transpose":
        if len(nodes) > 1:
            nodes[1].input[0] = nodes[0].input[0]
        nodes.pop(0)
    while nodes and nodes[-1].op_type == "Transpose":
        last_out = nodes[-1].input[0]
        del onnx_model.graph.output[:]
        vi = onnx.helper.make_tensor_value_info(last_out, onnx.TensorProto.FLOAT, None)
        onnx_model.graph.output.append(vi)
        nodes.pop()
    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(nodes)

    for inp in onnx_model.graph.input:
        if inp.name == "global_in":
            dims = inp.type.tensor_type.shape.dim
            dims[1].dim_value = 5   # H
            dims[2].dim_value = 5   # W
            dims[3].dim_value = 3   # C  (channels-last)

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)
    print("[3/6] Transpose nodes stripped — input fixed to NHWC (5,5,3)")

    # ── 4. Sanitize digit-leading initializer names ──────────────────────────
    # ONNX weight tensors sometimes get auto-generated names like "7xMmUO".
    # These start with a digit → invalid C identifiers.
    # hls4ml prefixes them with "constant" in the header but never emits
    # the nnet:: call in the .cpp → Vitis HLS sees an empty dataflow region
    # and aborts after ~95 min with "optimized away due to absence of outputs".
    # Fix: rename to w_<original> everywhere in the graph before hls4ml sees it.
    rename_map = {}
    for init in onnx_model.graph.initializer:
        if init.name and init.name[0].isdigit():
            new_name = "w_" + init.name
            rename_map[init.name] = new_name
            init.name = new_name
    if rename_map:
        print(f"[4/6] Sanitized {len(rename_map)} digit-leading initializer name(s): "
              f"{list(rename_map.keys())}")
        for node in onnx_model.graph.node:
            node.input[:] = [rename_map.get(x, x) for x in node.input]
        for vi in (list(onnx_model.graph.value_info) +
                   list(onnx_model.graph.input) +
                   list(onnx_model.graph.output)):
            if vi.name in rename_map:
                vi.name = rename_map[vi.name]
    else:
        print("[4/6] No digit-leading initializer names (clean ONNX)")

    for i, node in enumerate(onnx_model.graph.node):
        if not node.name:
            node.name = f"{node.op_type}_{i}"

    onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

    # ── 5. hls4ml config ────────────────────────────────────────────────────
    config = hls4ml.utils.config_from_onnx_model(
        onnx_model,
        granularity="name",
        backend=backend,
        default_precision=precision,
        default_reuse_factor=reuse_factor,
    )
    print(f"[5/6] Config: precision={precision}  reuse_factor={reuse_factor}  "
          f"clock={clock_period}ns ({1000/clock_period:.0f} MHz)  io={io_type}")

    # Patch hls4ml 1.3.0 bug: InferPrecisionTypes.match() calls
    # node.get_input_variable() which raises IndexError on Constant nodes
    # (weight tensors) whose inputs list is empty. Treat as no predecessor.
    from hls4ml.model.optimizer.passes.infer_precision import InferPrecisionTypes
    from hls4ml.model.types import UnspecifiedPrecisionType
    if not getattr(InferPrecisionTypes, '_patched_for_empty_inputs', False):
        def _safe_match(self, node):
            try:
                input_var = node.get_input_variable()
            except IndexError:
                input_var = None
            if input_var is not None and isinstance(input_var.type, UnspecifiedPrecisionType):
                return False
            for layer_type in node.types.values():
                if isinstance(layer_type.precision, UnspecifiedPrecisionType):
                    return True
            return False
        InferPrecisionTypes.match = _safe_match
        InferPrecisionTypes._patched_for_empty_inputs = True

    # ── 6. Convert → HLS C++ ────────────────────────────────────────────────
    conv_kwargs = dict(
        output_dir=output_dir,
        hls_config=config,
        backend=backend,
        clock_period=clock_period,
        io_type=io_type,
    )
    if board:
        conv_kwargs["board"] = board
    elif part:
        conv_kwargs["part"] = part

    hls_model = hls4ml.converters.convert_from_onnx_model(onnx_model, **conv_kwargs)
    hls_model.write()
    print(f"[6/6] HLS C++ written → {output_dir}/firmware/")
    print(f"  Target: {board or part or '(no device set)'}")
    return hls_model


# ─────────────────────────────────────────────────────────────────
# Report parsing
# ─────────────────────────────────────────────────────────────────

def parse_report(output_dir):
    """
    Parse myproject_csynth.rpt and return a structured metrics dict:
      clock_target_ns, clock_achieved_ns,
      latency_min_cycles, latency_max_cycles, II_min, II_max,
      resources: {BRAM_18K, DSP, FF, LUT, URAM} each with
                 {total, available, util_pct}
    """
    rpt_path = os.path.join(
        output_dir, "myproject_prj", "solution1",
        "syn", "report", "myproject_csynth.rpt"
    )
    if not os.path.exists(rpt_path):
        print(f"  Report not found: {rpt_path}")
        return None

    with open(rpt_path) as f:
        text = f.read()

    m = {}

    # ── Timing ──────────────────────────────────────────────────────────────
    tm = re.search(r'ap_clk\s+\|\s+([\d.]+)\s*ns\s*\|\s*([\d.]+)\s*ns', text)
    if tm:
        m["clock_target_ns"]   = float(tm.group(1))
        m["clock_achieved_ns"] = float(tm.group(2))

    # ── Latency (top-level summary row) ─────────────────────────────────────
    # Table row:  |  min  |  max  | abs_min | abs_max | II_min | II_max | Type |
    lat = re.search(
        r'\|\s*(\d+)\s*\|\s*(\d+)\s*\|[^|]+\|[^|]+\|\s*(\d+)\s*\|\s*(\d+)\s*\|',
        text
    )
    if lat:
        m["latency_min_cycles"] = int(lat.group(1))
        m["latency_max_cycles"] = int(lat.group(2))
        m["II_min"]             = int(lat.group(3))
        m["II_max"]             = int(lat.group(4))

    # ── Resources ───────────────────────────────────────────────────────────
    def _parse_row(label):
        rx = re.search(
            label + r'\s*\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|',
            text
        )
        if not rx:
            return None
        out = []
        for i in range(1, 6):
            raw = rx.group(i).strip().replace(',', '').replace('~', '').replace('%', '')
            try:
                out.append(int(raw))
            except ValueError:
                out.append(0)
        return out  # [BRAM_18K, DSP, FF, LUT, URAM]

    total = _parse_row(r'Total')
    avail = _parse_row(r'Available')
    util  = _parse_row(r'Utilization \(%\)')

    if total and avail and util:
        names = ["BRAM_18K", "DSP", "FF", "LUT", "URAM"]
        m["resources"] = {}
        for i, name in enumerate(names):
            m["resources"][name] = {
                "total":     total[i],
                "available": avail[i],
                "util_pct":  util[i],
            }

    return m


def print_metrics(metrics, clock_period_ns, teacher_params=368_000):
    """
    Print latency, II, throughput, resource table, and compression ratio.
    teacher_params: PhysFormerWrapper total parameters.
    """
    if not metrics:
        print("No metrics to display.")
        return

    clock_hz = 1e9 / clock_period_ns
    ii       = metrics.get("II_max", metrics.get("II_min", 1))
    lat_max  = metrics.get("latency_max_cycles", "?")
    lat_min  = metrics.get("latency_min_cycles", "?")
    ach_ns   = metrics.get("clock_achieved_ns", clock_period_ns)

    throughput_mhz = clock_hz / ii / 1e6
    lat_us         = lat_max * ach_ns / 1000 if isinstance(lat_max, int) else "?"

    student_params = 9_900
    compression    = teacher_params / student_params

    print("\n" + "=" * 62)
    print("  SYNTHESIS RESULTS — StudentNet / ZCU104")
    print("=" * 62)
    print(f"  Clock target   : {clock_period_ns} ns  ({clock_hz/1e6:.0f} MHz)")
    print(f"  Clock achieved : {ach_ns:.3f} ns  ({1e9/ach_ns:.0f} MHz max, slack {clock_period_ns-ach_ns:.3f} ns)")
    print(f"  Latency        : {lat_min}–{lat_max} cycles  ({lat_us} μs @ achieved clock)")
    print(f"  II             : {ii} cycles")
    print(f"  Throughput     : {throughput_mhz:.3f} M inf/s  "
          f"(1 inference / {ii * clock_period_ns / 1e3:.2f} μs)")
    print(f"  Compression    : {teacher_params/1000:.0f}K params → {student_params/1000:.1f}K params  "
          f"(×{compression:.0f})")
    print()
    print(f"  {'Resource':<12} {'Used':>12} {'Available':>12} {'Util %':>8}")
    print(f"  {'-'*48}")

    res = metrics.get("resources", {})
    fits = True
    for name in ["LUT", "FF", "DSP", "BRAM_18K", "URAM"]:
        r = res.get(name, {})
        pct  = r.get("util_pct", 0)
        flag = "  ⚠ OVER" if pct > 100 else ""
        if pct > 100:
            fits = False
        print(f"  {name:<12} {r.get('total', 0):>12,} {r.get('available', 0):>12,} "
              f"{pct:>7}%{flag}")

    print()
    if fits:
        print("  ✓ Design FITS on ZCU104 (xczu7ev-ffvc1156-2-e)")
    else:
        print("  ✗ Design DOES NOT FIT — increase --reuse-factor")
    print("=" * 62)

    return throughput_mhz


# ─────────────────────────────────────────────────────────────────
# Synthesis runner
# ─────────────────────────────────────────────────────────────────

def run_synthesis(hls_model, vitis_hls_bin=VITIS_HLS_BIN):
    """
    Run Vitis HLS C synthesis by calling vitis_hls directly.

    hls4ml 1.3.0 vivado backend hard-codes 'vivado_hls' (Vivado 2020 name).
    Vitis HLS 2023.1 installs only 'vitis_hls'. We bypass hls_model.build()
    and call vitis_hls ourselves via subprocess with the generated build_prj.tcl.

    Returns parsed metrics dict.
    """
    import subprocess

    out_dir = hls_model.config.get_output_dir()
    if " " in out_dir:
        raise ValueError(
            f"Output dir has spaces: '{out_dir}'\n"
            f"Vitis HLS cannot handle spaces — use --out /tmp/hls_pikd_rf16"
        )

    vitis_hls_exe = os.path.join(vitis_hls_bin, "vitis_hls")
    if not os.path.exists(vitis_hls_exe):
        raise FileNotFoundError(f"vitis_hls not found at: {vitis_hls_exe}")

    tcl = os.path.join(out_dir, "build_prj.tcl")
    if not os.path.exists(tcl):
        raise FileNotFoundError(f"build_prj.tcl not found: {tcl}")

    cmd = [vitis_hls_exe, "-f", "build_prj.tcl",
           "reset=1", "csim=0", "synth=1", "cosim=0", "validation=0", "export=0"]

    print(f"\nRunning HLS C synthesis in: {out_dir}")
    print(f"  Command: {' '.join(cmd)}")
    print("  (C synthesis only — no cosim/export — ~10-30 min with rf=16)")

    log_path = out_dir + "_synth.log"
    with open(log_path, "w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=out_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_f.write(proc.stdout)

    # Also print the last 40 lines so user sees completion/errors
    lines = proc.stdout.strip().splitlines()
    print("\n--- Vitis HLS output (last 40 lines) ---")
    for line in lines[-40:]:
        print(f"  {line}")
    print(f"--- Full log: {log_path} ---")

    if proc.returncode != 0:
        print(f"WARNING: vitis_hls exited with code {proc.returncode}")

    metrics = parse_report(out_dir)
    return metrics


# ─────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    BASE         = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    DEFAULT_CKPT = os.path.join(BASE, "checkpoints_qat", "best_qat.pt")
    # Default out: /tmp to avoid spaces (Vitis HLS requirement)
    DEFAULT_OUT  = "/tmp/hls_pikd_rf16"

    parser = argparse.ArgumentParser(
        description="Export QAT StudentNet → HLS (ONNX backend, ZCU104 target)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt",         default=DEFAULT_CKPT,
                        help="QAT checkpoint path (best_qat.pt)")
    parser.add_argument("--out",          default=DEFAULT_OUT,
                        help="HLS output dir (NO SPACES in path)")
    parser.add_argument("--backend",      default="Vivado")
    parser.add_argument("--part",         default="xczu7ev-ffvc1156-2-e",
                        help="ZCU104 part number")
    parser.add_argument("--board",        default=None,
                        help="Board name (overrides --part if set)")
    parser.add_argument("--clock",        default=5.0, type=float,
                        help="Clock period in ns (5.0 = 200 MHz)")
    parser.add_argument("--precision",    default="fixed<8,4>",
                        help="hls4ml precision (fixed<8,4> = ap_fixed<8,4>)")
    parser.add_argument("--reuse-factor", default=16, type=int, dest="reuse_factor",
                        help="16 fits ZCU104; 4 fits VU9P with better throughput")
    parser.add_argument("--io-type",      default="io_parallel", dest="io_type",
                        help="io_parallel (lowest latency) or io_stream")
    parser.add_argument("--teacher-params", default=368_000, type=int, dest="teacher_params",
                        help="Teacher parameter count for compression ratio")
    parser.add_argument("--profile",      action="store_true",
                        help="Run PyTorch activation profiling before export")
    parser.add_argument("--synth",        action="store_true",
                        help="Run HLS synthesis after code generation")
    parser.add_argument("--copy-back",    default=None, dest="copy_back",
                        help="Copy HLS output back to this project dir after synthesis")
    args = parser.parse_args()

    print("=" * 62)
    print("  StudentNet → hls4ml  (ONNX backend)")
    print(f"  Target: {args.part or args.board}  |  rf={args.reuse_factor}  |  "
          f"{1000/args.clock:.0f} MHz  |  {args.precision}")
    print("=" * 62)

    # ── Load model ──────────────────────────────────────────────────────────
    model, cfg = load_for_hls4ml(args.ckpt)
    model      = replace_relu6(model)
    wrapper    = HLS4MLWrapper(model)
    wrapper.eval()

    with torch.no_grad():
        test_out = wrapper(torch.randn(1, 3, 5, 5))
    print(f"Wrapper output: {tuple(test_out.shape)}  (expect (1, 1, 5, 5))")

    # ── Activation profiling ─────────────────────────────────────────────────
    if args.profile:
        # Parse integer bits from precision string: fixed<8,4> → 4
        try:
            int_bits = int(args.precision.split(",")[1].replace(">", "").strip())
            tot_bits = int(args.precision.split("<")[1].split(",")[0].strip())
        except Exception:
            int_bits, tot_bits = 4, 8
        profile_activations(wrapper, precision_int_bits=int_bits, total_bits=tot_bits)

    # ── Export ───────────────────────────────────────────────────────────────
    hls_model = export_to_hls(
        wrapper,
        output_dir=args.out,
        backend=args.backend,
        part=args.part if not args.board else None,
        board=args.board,
        clock_period=args.clock,
        io_type=args.io_type,
        reuse_factor=args.reuse_factor,
        precision=args.precision,
    )

    if hls_model is None:
        return

    # ── Synthesis ────────────────────────────────────────────────────────────
    if args.synth:
        if not (args.part or args.board):
            print("ERROR: --synth requires --part or --board")
            return
        metrics = run_synthesis(hls_model)
        if metrics:
            print_metrics(metrics, args.clock, teacher_params=args.teacher_params)

    # ── Copy results back to project dir ─────────────────────────────────────
    if args.copy_back:
        dest = os.path.join(args.copy_back, "hls_output_rf16")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(args.out, dest)
        print(f"\nResults copied → {dest}")

    print("\nDone.")
    if not args.synth:
        print("To run synthesis:")
        print(f"  export PATH={VITIS_HLS_BIN}:$PATH")
        print(f"  python -m pi_kd.quantization.export_hls4ml --synth --out {args.out}")


if __name__ == "__main__":
    main()
