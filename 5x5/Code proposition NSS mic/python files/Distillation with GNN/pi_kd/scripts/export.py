#!/usr/bin/env python3
"""
Export pipeline: QAT fine-tuning -> quantized model -> hls4ml export.

Usage:
  python -m pi_kd.scripts.export --ckpt checkpoints_pikd/best.pt
  python -m pi_kd.scripts.export --ckpt checkpoints_pikd/best.pt --skip-qat --onnx-only

Run from the parent directory (Distillation with GNN/).
"""

import os
import sys
import argparse
import yaml
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIKD_DIR = os.path.dirname(SCRIPT_DIR)
PARENT_DIR = os.path.dirname(PIKD_DIR)
for d in [PIKD_DIR, PARENT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pi_kd.data.datamodule import load_data
from pi_kd.evaluation.metrics import compute_all_metrics
from pi_kd.quantization.qat import prepare_for_qat, qat_finetune, convert_and_save
from pi_kd.quantization.export_hls4ml import export_to_hls, export_onnx


def main():
    parser = argparse.ArgumentParser(description="PI-KD Export Pipeline")
    parser.add_argument("--ckpt", required=True, help="Best student checkpoint (float32)")
    parser.add_argument("--config", default=os.path.join(PIKD_DIR, "configs", "train_config.yaml"))
    parser.add_argument("--skip-qat", action="store_true", help="Skip QAT, export float32 directly")
    parser.add_argument("--onnx-only", action="store_true", help="Only export ONNX (no hls4ml)")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dl_train, dl_val, _ = load_data(cfg)

    if not args.skip_qat:
        # ── QAT ──────────────────────────────────
        print("\n[1/3] Preparing for QAT...")
        model, _ = prepare_for_qat(None, args.ckpt, device)

        # Evaluate float32 baseline
        print("\n[2/3] Float32 baseline metrics:")
        f32_metrics = compute_all_metrics(model, dl_val, device)
        f32_f1 = f32_metrics.get("composite_f1", 0)
        print(f"  Composite F1 (float32): {f32_f1:.4f}")

        qat_cfg = cfg.get("qat", {})
        print(f"\n[3/3] QAT fine-tuning ({qat_cfg.get('epochs', 20)} epochs)...")
        model, best_score = qat_finetune(
            model, dl_train, dl_val, cfg,
            device=device,
            epochs=qat_cfg.get("epochs", 20),
            ckpt_dir=qat_cfg.get("ckpt_dir", "./checkpoints_qat"),
        )

        # Convert and check degradation
        model_q = convert_and_save(model, "student_quantized.pt")
        q_metrics = compute_all_metrics(model_q, dl_val, device)
        q_f1 = q_metrics.get("composite_f1", 0)
        degradation = f32_f1 - q_f1
        print(f"\n  F1 degradation: {degradation:.4f} ({'PASS' if degradation < 0.01 else 'FAIL'} < 1%)")

        export_model = model_q
    else:
        from pi_kd.evaluation.compare import load_student_from_ckpt
        export_model = load_student_from_ckpt(args.ckpt, device)

    # ── Export ────────────────────────────────
    if args.onnx_only:
        export_onnx(export_model, "student_export.onnx")
    else:
        export_to_hls(export_model, output_dir="hls_output")


if __name__ == "__main__":
    main()
