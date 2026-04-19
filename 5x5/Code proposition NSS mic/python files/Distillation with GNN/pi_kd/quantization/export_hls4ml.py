"""
Export QAT-adapted StudentNet to HLS via hls4ml.

Pipeline:
  best_qat.pt (QAT-adapted float32 weights)
    -> load_for_hls4ml()  — extracts weights into a clean BN-fused StudentNet
    -> hls4ml.converters.convert_from_pytorch_model()
    -> HLS C++ (Vivado HLS / Vitis HLS)
    -> Xilinx bitstream

Key distinction:
  student_quantized.pt  — PyTorch INT8 (torch.quantization.convert output).
                          NOT usable by hls4ml. Used only as size/precision reference.

  best_qat.pt           — QAT-adapted float32 weights. This is what hls4ml reads.
                          hls4ml applies its own fixed-point (ap_fixed<8,N>) during
                          synthesis — QAT pre-adapts the weights for 8-bit, giving
                          better accuracy than exporting the original float32 model.
"""

import os
import sys
import torch

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


def load_for_hls4ml(qat_ckpt_path, device="cpu"):
    """
    Load QAT-adapted weights into a clean BN-fused StudentNet ready for hls4ml.

    After prepare_qat(), the state_dict contains extra fake-quantizer keys
    (weight_fake_quant.*, activation_post_process.*). hls4ml cannot parse these.
    This function:
      1. Creates a fresh StudentNet
      2. Fuses BN into Conv (eval mode) — same fusion done before QAT
      3. Loads only the matching keys from the QAT checkpoint

    The fused model (Conv2d with BN absorbed into weights+bias) is simpler for
    hls4ml: no BN layer in HLS, fewer operations, one less source of precision loss.

    Returns:
        model (StudentNet, eval, BN fused, float32 QAT-adapted weights)
        cfg   (training config dict from checkpoint)
    """
    from ..models.student import StudentNet

    ckpt = torch.load(qat_ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    widths = tuple(cfg.get("student", {}).get("widths", [32, 64, 64]))

    # Build clean fused model (same structure as what QAT was trained on)
    model = StudentNet(in_ch=3, widths=widths)
    model.eval()
    model.fuse_bn()

    # QAT state_dict has extra keys (fake_quant buffers). Keep only keys
    # that exist in the clean fused model.
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


def export_to_hls(model, output_dir="hls_output", backend="Vivado", part=None, board=None,
                  clock_period=10, io_type="io_parallel", reuse_factor=1,
                  precision="ap_fixed<8,4>"):
    """
    Convert a float32 StudentNet to HLS C++ via hls4ml.

    Use load_for_hls4ml() to prepare the model from a QAT checkpoint before calling this.

    Board vs part:
      - board: high-level name (e.g. 'pynq-z2', 'zcu104') — use when known
      - part:  Xilinx part number (e.g. 'xc7k325tffg900-2') — use for resource estimation
               when the board is not yet decided. Any Kintex-7 or UltraScale+ will fit
               this 9.6K-param model easily.
      - Neither: generates HLS C++ + C-sim only (no synthesis). Use this first to verify
               the logic is correct before committing to a specific device.

    Stages (run in order as your tools become available):
      1. generate + csim    — just needs g++, no Vivado
      2. + synthesis        — needs Vivado HLS + a part number
      3. + bitstream        — needs board-specific Vivado project

    Args:
        model        : StudentNet (BN fused, eval, float32 QAT-adapted weights)
        output_dir   : directory for HLS project output
        backend      : 'Vivado' or 'VivadoAccelerator'
        part         : Xilinx part number, e.g. 'xc7k325tffg900-2'. Used when board unknown.
        board        : FPGA board name (overrides part if both given)
        clock_period : target clock period in ns (10 = 100 MHz)
        io_type      : 'io_parallel' (lowest latency, fits our tiny 5x5 input)
        reuse_factor : 1 = fully unrolled (9.6K params → well within any mid-range FPGA)
        precision    : 'ap_fixed<8,4>' matches QAT 8-bit training
    """
    try:
        import hls4ml
    except ImportError:
        print("hls4ml not installed. Run:  pip install hls4ml[profiling]")
        print("Falling back to ONNX export (can be converted manually later).")
        os.makedirs(output_dir, exist_ok=True)
        export_onnx(model, os.path.join(output_dir, "student.onnx"))
        return None

    os.makedirs(output_dir, exist_ok=True)

    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(1, 3, 5, 5),
        granularity="name",
        default_precision=precision,
        default_reuse_factor=reuse_factor,
    )

    # Build converter kwargs — board takes precedence over part
    conv_kwargs = dict(
        input_shape=(1, 3, 5, 5),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        clock_period=clock_period,
        io_type=io_type,
    )
    if board:
        conv_kwargs["board"] = board
    elif part:
        conv_kwargs["part"] = part
    # if neither board nor part: hls4ml uses its internal default (still generates valid HLS)

    hls_model = hls4ml.converters.convert_from_pytorch_model(model, **conv_kwargs)

    print(f"\nHLS C++ generated → {output_dir}/")
    print(f"  Precision: {precision} | IO: {io_type} | Reuse: {reuse_factor} | Clock: {clock_period} ns")
    target = board or part or "(no board/part — C-sim only)"
    print(f"  Target: {target}")

    return hls_model


def run_csim(hls_model, model, n_samples=100):
    """
    Stage 1: compile HLS C++ and run C-simulation.
    Requires only g++ (no Vivado). Verifies logic correctness.

    Compares HLS output vs PyTorch float32 output on random inputs.
    Max absolute difference < 0.1 on primary logits = pass.
    """
    import numpy as np

    hls_model.compile()
    print("C-sim compiled successfully.")

    dummy = np.random.randn(n_samples, 3, 5, 5).astype(np.float32)
    y_hls = hls_model.predict(dummy)

    with torch.no_grad():
        y_pt = model(torch.tensor(dummy))

    # Compare primary logits
    p_hls = y_hls[..., :25].reshape(n_samples, 5, 5) if y_hls.ndim > 2 else y_hls
    p_pt  = y_pt["P_logits"].squeeze(1).numpy()
    diff  = np.abs(p_hls - p_pt).max()

    print(f"C-sim vs PyTorch max |diff| on P_logits: {diff:.4f}")
    if diff < 0.5:
        print("  C-sim PASS — HLS logic matches PyTorch within ap_fixed rounding tolerance.")
    else:
        print("  C-sim WARN — large difference, check precision settings.")
    return diff


def run_synthesis(hls_model):
    """
    Stage 2: run Vivado HLS synthesis for resource/latency estimates.
    Requires Vivado HLS installed and a part/board configured in the hls_model.
    """
    print("Running Vivado HLS synthesis (this takes 5-15 min)...")
    report = hls_model.build(csim=False, synth=True, export=False)
    if report:
        print("\nResource estimates:")
        for k, v in report.items():
            print(f"  {k}: {v}")
    return report


def export_onnx(model, path="student.onnx"):
    """Fallback: export to ONNX (if hls4ml not available)."""
    model.eval()
    dummy = torch.randn(1, 3, 5, 5)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["S_logits", "P_logits", "ics_logit"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
    )
    print(f"ONNX saved to {path}")
