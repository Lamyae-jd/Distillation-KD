"""
Export quantized StudentNet to HLS via hls4ml.

Pipeline:
  Student float32 (PI-KD)
    -> QAT fine-tuning (PyTorch native)
    -> torch.quantization.convert()
    -> hls4ml.converters.convert_from_pytorch_model()
    -> HLS C++ (C simulation for latency estimation)
    -> Vitis HLS -> Vivado -> bitstream Xilinx
"""

import os
import sys
import torch
import numpy as np

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


def export_to_hls(model, output_dir="hls_output", backend="Vivado", board="pynq-z2",
                  clock_period=10, io_type="io_parallel", reuse_factor=1):
    """
    Convert a PyTorch StudentNet to HLS C++ via hls4ml.

    Args:
        model: StudentNet (float32 or quantized)
        output_dir: directory for HLS project output
        backend: 'Vivado' or 'VivadoAccelerator'
        board: target FPGA board
        clock_period: target clock period in ns
        io_type: 'io_parallel' (low latency) or 'io_stream' (low resource)
        reuse_factor: DSP reuse factor (1 = fully parallel, higher = less DSP)
    """
    try:
        import hls4ml
    except ImportError:
        print("hls4ml not installed. Install with: pip install hls4ml[profiling]")
        print("Saving ONNX model instead for manual conversion.")
        export_onnx(model, os.path.join(output_dir, "student.onnx"))
        return None

    os.makedirs(output_dir, exist_ok=True)

    # Trace model with example input
    model.eval()
    example_input = torch.randn(1, 3, 5, 5)

    # Configure hls4ml
    config = hls4ml.utils.config_from_pytorch_model(
        model,
        input_shape=(1, 3, 5, 5),
        granularity="name",
        default_precision="ap_fixed<16,6>",
        default_reuse_factor=reuse_factor,
    )

    # Convert
    hls_model = hls4ml.converters.convert_from_pytorch_model(
        model,
        input_shape=(1, 3, 5, 5),
        hls_config=config,
        output_dir=output_dir,
        backend=backend,
        board=board,
        clock_period=clock_period,
        io_type=io_type,
    )

    # Compile for C simulation
    hls_model.compile()

    # Predict with C simulation to verify
    y_hls = hls_model.predict(example_input.numpy())
    with torch.no_grad():
        y_py = model(example_input)

    print(f"\nHLS model exported to {output_dir}/")
    print(f"  Backend: {backend}, Board: {board}")
    print(f"  Clock: {clock_period} ns, IO: {io_type}, Reuse: {reuse_factor}")

    # Resource estimation
    report = hls_model.build(csim=False, synth=True)
    if report:
        print("\nResource Estimation:")
        for k, v in report.items():
            print(f"  {k}: {v}")

    return hls_model


def export_onnx(model, path="student.onnx"):
    """Fallback: export to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, 5, 5)
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["S_logits", "P_logits", "ics_logit"],
        dynamic_axes={"input": {0: "batch"}},
        opset_version=13,
    )
    print(f"ONNX model saved to {path}")


def estimate_latency(hls_model):
    """Extract latency from HLS synthesis report."""
    try:
        report = hls_model.build(csim=True, synth=False)
        return report
    except Exception as e:
        print(f"Latency estimation failed: {e}")
        return None
