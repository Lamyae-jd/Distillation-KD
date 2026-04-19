#!/usr/bin/env python3
"""
Evaluate a trained student model or run the full ablation study.

Usage:
  python -m pi_kd.scripts.evaluate --ckpt checkpoints_pikd/best.pt
  python -m pi_kd.scripts.evaluate --ablation

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
from pi_kd.evaluation.metrics import compute_all_metrics, model_stats, print_comparison
from pi_kd.evaluation.compare import run_ablation, load_student_from_ckpt, load_teacher_wrapper


def main():
    parser = argparse.ArgumentParser(description="PI-KD Evaluation")
    parser.add_argument("--config", default=os.path.join(PIKD_DIR, "configs", "train_config.yaml"))
    parser.add_argument("--ckpt", default=None, help="Student checkpoint to evaluate")
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    _, dl_val, _ = load_data(cfg, verbose=True)

    if args.ablation:
        results = run_ablation(
            dl_val,
            teacher_ckpt=os.path.join(PARENT_DIR, "checkpoints", "best.pt"),
            student_no_kd_ckpt=os.path.join(PIKD_DIR, "checkpoints_noKD", "best.pt"),
            student_kd_ckpt=os.path.join(PIKD_DIR, "checkpoints_KD", "best.pt"),
            student_pikd_ckpt=os.path.join(PIKD_DIR, "checkpoints_pikd", "best.pt"),
            device=device,
        )
    elif args.ckpt:
        print(f"\nEvaluating {args.ckpt}...")
        student = load_student_from_ckpt(args.ckpt, device)
        metrics = compute_all_metrics(student, dl_val, device)
        stats = model_stats(student)
        print("\nResults:")
        print_comparison({"Student": {**metrics, **stats}})
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
