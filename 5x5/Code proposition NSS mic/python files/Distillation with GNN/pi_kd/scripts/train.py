#!/usr/bin/env python3
"""
PI-KD Training Script
=====================
Physics-Informed Knowledge Distillation for FPGA-deployable ICS detection.

Usage:
  python -m pi_kd.scripts.train                          # full PI-KD
  python -m pi_kd.scripts.train --no-teacher              # ablation: no KD
  python -m pi_kd.scripts.train --no-physics              # ablation: KD without physics
  python -m pi_kd.scripts.train --no-feat                 # ablation: no feature matching
  python -m pi_kd.scripts.train --config path/to/cfg.yaml # custom config
  python -m pi_kd.scripts.train --tag v2_adaptive_phys    # custom run tag
  python -m pi_kd.scripts.train --qat                     # QAT only (skip distillation)
  python -m pi_kd.scripts.train --qat --ckpt checkpoints_pikd/best.pt  # QAT from specific ckpt

Run from the parent directory (Distillation with GNN/).
"""

import os
import sys
import random
import argparse
from datetime import datetime

import numpy as np
import torch
import yaml

# Ensure parent directory is on path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIKD_DIR = os.path.dirname(SCRIPT_DIR)
PARENT_DIR = os.path.dirname(PIKD_DIR)
for d in [PIKD_DIR, PARENT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pi_kd.models.student import StudentNet
from pi_kd.models.teacher import load_teacher, load_multi_teacher
from pi_kd.data.datamodule import load_data
from pi_kd.training.trainer import PIKDTrainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="PI-KD Training")
    parser.add_argument("--config", default=os.path.join(PIKD_DIR, "configs", "train_config.yaml"))
    parser.add_argument("--no-teacher", action="store_true", help="Ablation: train without KD")
    parser.add_argument("--no-physics", action="store_true", help="Ablation: KD without physics loss")
    parser.add_argument("--no-feat", action="store_true", help="Ablation: no feature matching")
    parser.add_argument("--device", default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--resume", default=None, metavar="CKPT",
                        help="Resume training from a checkpoint (e.g. checkpoints_pikd/best.pt). "
                             "Loads student, optimizer, scaler state and starts from the saved epoch.")
    parser.add_argument("--init-from", default=None, metavar="CKPT", dest="init_from",
                        help="Initialize student weights from a checkpoint with strict=False, "
                             "but start fresh optimizer/scaler/scheduler (epoch 0). "
                             "Used to fine-tune a mono-head model from an old multi-head checkpoint.")
    parser.add_argument("--tag", default=None,
                        help="Optional run tag appended to log filename (e.g. 'v2_adaptive_phys')")
    parser.add_argument("--qat", action="store_true",
                        help="Run Quantization-Aware Training only (skip distillation)")
    parser.add_argument("--ckpt", default=None, metavar="PATH",
                        help="Checkpoint path for QAT (default: checkpoints_pikd/best.pt)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # ── Auto-versioned log file ──────────────────────
    os.makedirs(os.path.join(PARENT_DIR, "logs"), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    log_filename = f"logs/run_{timestamp}{tag}.log"
    log_path = os.path.join(PARENT_DIR, log_filename)

    import logging
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)

    class TeeWriter:
        """Write to both stdout and a log file simultaneously."""
        def __init__(self, log_path):
            self._file = open(log_path, "w")
            self._stdout = sys.stdout
        def write(self, msg):
            self._stdout.write(msg)
            self._file.write(msg)
            self._file.flush()
        def flush(self):
            self._stdout.flush()
            self._file.flush()

    sys.stdout = TeeWriter(log_path)

    set_seed(cfg["train"]["seed"])
    print(f"Run log: {log_filename}")
    print(f"Device: {device}")
    print(f"Config: {args.config}")

    # ── QAT-only mode ─────────────────────────────
    if args.qat:
        from pi_kd.quantization.qat import prepare_for_qat, qat_finetune, convert_and_save

        ckpt_path = args.ckpt or os.path.join(PARENT_DIR, "checkpoints_pikd", "best.pt")
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PARENT_DIR, ckpt_path)
        print(f"\n[QAT] Loading float32 checkpoint: {ckpt_path}")

        print("\n[1/4] Loading data...")
        dl_train, dl_val, _ = load_data(cfg)

        print("\n[2/4] Preparing model for QAT...")
        model_qat, qat_cfg = prepare_for_qat(None, ckpt_path, device)
        qat_cfg.update(cfg)

        print(f"\n[3/4] QAT fine-tuning...")
        qat_epochs = cfg.get("qat", {}).get("epochs", 20)
        qat_ckpt_dir = cfg.get("qat", {}).get("ckpt_dir", "./checkpoints_qat")
        qat_ckpt_dir = os.path.join(PARENT_DIR, qat_ckpt_dir) if not os.path.isabs(qat_ckpt_dir) else qat_ckpt_dir

        model_qat, best_metrics = qat_finetune(
            model_qat, dl_train, dl_val, qat_cfg,
            device=device, epochs=qat_epochs, ckpt_dir=qat_ckpt_dir,
        )

        print(f"\n[4/4] Converting to INT8...")
        q_path = os.path.join(qat_ckpt_dir, "student_quantized.pt")
        convert_and_save(model_qat, q_path)

        print(f"\nQAT pipeline complete.")
        return

    # Override for ablations
    if args.no_teacher:
        cfg["distillation"]["lambda_kd"] = 0.0
        cfg["distillation"]["lambda_phys"] = 0.0
        cfg["distillation"]["lambda_feat"] = 0.0
        cfg["distillation"]["use_feature_matching"] = False
        print("ABLATION: No teacher (task loss only)")
    if args.no_physics:
        cfg["distillation"]["lambda_phys"] = 0.0
        print("ABLATION: No physics loss")
    if args.no_feat:
        cfg["distillation"]["use_feature_matching"] = False
        cfg["distillation"]["lambda_feat"] = 0.0
        print("ABLATION: No feature matching")

    # ── 1. Data ──────────────────────────────────
    print("\n[1/3] Loading data...")
    dl_train, dl_val, _ = load_data(cfg)

    # ── 2. Models ────────────────────────────────
    print("\n[2/3] Building models...")

    # Student
    s_cfg = cfg.get("student", {})
    student = StudentNet(
        in_ch=s_cfg.get("in_ch", 3),
        widths=tuple(s_cfg.get("widths", [32, 64, 64])),
    ).to(device)
    print(f"  Student: {student.count_params():,} params, ~{student.count_flops():,} FLOPs")

    # Teacher(s) — single or multi-teacher ensemble (A)
    teachers = None
    kd_cfg = cfg.get("distillation", {})
    if kd_cfg.get("lambda_kd", 0) > 0 or kd_cfg.get("lambda_feat", 0) > 0:
        teacher_cfg = cfg.get("teacher", {})
        ckpt_list = teacher_cfg.get("checkpoints", [])

        # Resolve paths relative to pi_kd directory
        resolved = []
        for p in ckpt_list:
            full = os.path.join(PIKD_DIR, p) if not os.path.isabs(p) else p
            if os.path.exists(full):
                resolved.append(full)
            else:
                alt = os.path.join(PARENT_DIR, "checkpoints", os.path.basename(p))
                if os.path.exists(alt):
                    resolved.append(alt)
                else:
                    print(f"  WARNING: checkpoint not found: {p}")

        if len(resolved) == 0:
            # Fallback: single checkpoint key
            single = teacher_cfg.get("checkpoint", "")
            full = os.path.join(PIKD_DIR, single) if single else ""
            if os.path.exists(full):
                resolved = [full]
            else:
                alt = os.path.join(PARENT_DIR, "checkpoints", "best.pt")
                if os.path.exists(alt):
                    resolved = [alt]

        if len(resolved) > 1:
            print(f"  Loading {len(resolved)} teachers (multi-teacher ensemble)...")
            teachers = load_multi_teacher(resolved, device)
        elif len(resolved) == 1:
            print(f"  Loading single teacher from {resolved[0]}")
            teachers = [load_teacher(resolved[0], device)]
        else:
            print("  WARNING: no teacher checkpoints found, running without KD")
            cfg["distillation"]["lambda_kd"] = 0.0
            cfg["distillation"]["lambda_feat"] = 0.0

        if teachers:
            t_params = sum(p.numel() for p in teachers[0].parameters())
            print(f"  Teacher: {t_params:,} params (frozen)")
            print(f"  Compression ratio: {student.count_params() / t_params:.1%}")
    else:
        print("  No teacher (ablation mode)")

    # ── 3. Train ─────────────────────────────────
    print("\n[3/3] Training...")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    trainer = PIKDTrainer(
        student=student,
        teacher=teachers,  # list or None
        optimizer=optimizer,
        cfg=cfg,
        device=device,
        ckpt_dir=cfg.get("ckpt_dir", "./checkpoints_pikd"),
    )

    # Add feature matching projection params to optimizer
    if trainer.use_feature_matching and teachers:
        trainer.opt.add_param_group({
            "params": list(trainer.feat_loss.parameters()),
            "lr": cfg["train"]["lr"],
        })

    start_epoch = 0
    if args.resume:
        ckpt_path = args.resume
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PARENT_DIR, ckpt_path)
        print(f"\n[Resume] Loading checkpoint from: {ckpt_path}")
        saved_epoch = trainer.load_checkpoint(ckpt_path)
        start_epoch = saved_epoch + 1
        print(f"[Resume] Starting from epoch {start_epoch}")
    elif args.init_from:
        ckpt_path = args.init_from
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(PARENT_DIR, ckpt_path)
        print(f"\n[Init-from] Loading student weights from: {ckpt_path}")
        trainer.load_checkpoint(ckpt_path, weights_only=True)
        start_epoch = 0
        print(f"[Init-from] Starting fresh optimizer/scheduler at epoch 0")

    trainer.fit(
        dl_train, dl_val,
        epochs=cfg["train"]["epochs"],
        early_patience=cfg["train"].get("early_patience", 7),
        start_epoch=start_epoch,
    )

    # Generate figures from this run's log
    try:
        from pi_kd.scripts.plot_training import generate_figures
        fig_dir = os.path.join(PARENT_DIR, "figures", timestamp + tag)
        generate_figures(log_path, fig_dir)
        print(f"\nFigures saved to {fig_dir}/")
    except Exception as e:
        print(f"\nWARNING: Could not generate figures: {e}")


if __name__ == "__main__":
    main()
