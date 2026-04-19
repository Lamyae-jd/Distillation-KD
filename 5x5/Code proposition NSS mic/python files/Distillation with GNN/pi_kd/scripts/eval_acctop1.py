#!/usr/bin/env python3
"""
Quick evaluation: compute AccTop1 for the student (and optionally the teacher)
on the validation set, using the same data pipeline as training.

Usage:
  python -m pi_kd.scripts.eval_acctop1
  python -m pi_kd.scripts.eval_acctop1 --ckpt checkpoints_pikd/best.pt
  python -m pi_kd.scripts.eval_acctop1 --with-teacher
"""

import os
import sys
import argparse

import numpy as np
import torch
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIKD_DIR = os.path.dirname(SCRIPT_DIR)
PARENT_DIR = os.path.dirname(PIKD_DIR)
for d in [PIKD_DIR, PARENT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

from pi_kd.models.student import StudentNet
from pi_kd.models.teacher import load_teacher
from pi_kd.data.datamodule import load_data


def compute_acctop1_student(model, loader, device):
    """
    Student AccTop1: argmax of P_logits over the 5x5 grid,
    check if that pixel is a true primary.
    """
    model.eval()
    top1_correct = 0
    top1_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            P_star = y["P_star"].cpu()  # (B, 5, 5)
            B = x.shape[0]

            out = model(x)
            P_logits = out["P_logits"].squeeze(1).cpu()  # (B, 5, 5)

            P_flat = P_logits.view(B, -1)       # (B, 25)
            GT_flat = P_star.view(B, -1)         # (B, 25)

            pred_idx = P_flat.argmax(dim=1)      # (B,)
            has_primary = GT_flat.sum(dim=1) > 0

            gt_at_pred = GT_flat[torch.arange(B), pred_idx]
            top1_correct += int((gt_at_pred[has_primary] > 0).sum())
            top1_total += int(has_primary.sum())

    acc = top1_correct / top1_total if top1_total > 0 else float("nan")
    return acc, top1_correct, top1_total


def compute_acctop1_teacher(model, loader, device):
    """
    Teacher AccTop1: uses primary_idx (K candidates from GraphPrimaryHead)
    and primary_logits_raw (raw logits, same as original trainer).
    Passes T_star to the teacher for correct PhysicsBiasedMHSA attention.
    """
    model.eval()
    top1_correct = 0
    top1_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            T_star = y["T_star"].unsqueeze(1).to(device)  # (B,1,5,5)
            P_star = y["P_star"].cpu()
            B = x.shape[0]

            out = model(x, use_T_bias=T_star)
            logitsK = out["primary_logits_raw"].cpu()  # (B, K) raw logits
            idxK = out["primary_idx"].cpu()

            GT_flat = P_star.view(B, -1)

            best_k = logitsK.argmax(dim=1)
            pred_pixel = idxK.gather(1, best_k.unsqueeze(1)).squeeze(1)

            has_primary = GT_flat.sum(dim=1) > 0

            gt_at_pred = GT_flat[torch.arange(B), pred_pixel]
            top1_correct += int((gt_at_pred[has_primary] > 0).sum())
            top1_total += int(has_primary.sum())

    acc = top1_correct / top1_total if top1_total > 0 else float("nan")
    return acc, top1_correct, top1_total


def compute_acctop1_teacher_grid(model, loader, device):
    """
    Teacher AccTop1 on the 5x5 grid (argmax of P_logits),
    with T_star passed for correct MHSA attention.
    """
    model.eval()
    top1_correct = 0
    top1_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            T_star = y["T_star"].unsqueeze(1).to(device)
            P_star = y["P_star"].cpu()
            B = x.shape[0]

            out = model(x, use_T_bias=T_star)
            P_logits = out["P_logits"].squeeze(1).cpu()

            P_flat = P_logits.view(B, -1)
            GT_flat = P_star.view(B, -1)

            pred_idx = P_flat.argmax(dim=1)
            has_primary = GT_flat.sum(dim=1) > 0

            gt_at_pred = GT_flat[torch.arange(B), pred_idx]
            top1_correct += int((gt_at_pred[has_primary] > 0).sum())
            top1_total += int(has_primary.sum())

    acc = top1_correct / top1_total if top1_total > 0 else float("nan")
    return acc, top1_correct, top1_total


def main():
    parser = argparse.ArgumentParser(description="Eval AccTop1")
    parser.add_argument("--config", default=os.path.join(PIKD_DIR, "configs", "train_config.yaml"))
    parser.add_argument("--ckpt", default="checkpoints_pikd/best.pt",
                        help="Student checkpoint path (relative to project root)")
    parser.add_argument("--with-teacher", action="store_true",
                        help="Also evaluate the teacher for comparison")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device if torch.cuda.is_available() else "cpu"

    print("Loading data...")
    _, dl_val, _ = load_data(cfg)
    print(f"  Val batches: {len(dl_val)}")

    # Student
    ckpt_path = args.ckpt if os.path.isabs(args.ckpt) else os.path.join(PARENT_DIR, args.ckpt)
    print(f"\nLoading student from {ckpt_path} ...")
    s_cfg = cfg.get("student", {})
    student = StudentNet(
        in_ch=s_cfg.get("in_ch", 3),
        widths=tuple(s_cfg.get("widths", [32, 64, 64])),
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    student.load_state_dict(state["student"])
    print(f"  Loaded (epoch {state.get('epoch', '?')})")

    print("\nEvaluating student AccTop1 (argmax over 5x5 grid) ...")
    acc, correct, total = compute_acctop1_student(student, dl_val, device)
    print(f"  Student AccTop1 = {acc:.4f}  ({correct}/{total})")

    # Teacher (optional)
    if args.with_teacher:
        teacher_cfg = cfg.get("teacher", {})
        ckpt_list = teacher_cfg.get("checkpoints", [])
        t_path = ckpt_list[0] if ckpt_list else "../checkpoints/best.pt"
        t_full = os.path.join(PIKD_DIR, t_path) if not os.path.isabs(t_path) else t_path
        if not os.path.exists(t_full):
            t_full = os.path.join(PARENT_DIR, "checkpoints", "best.pt")

        print(f"\nLoading teacher from {t_full} ...")
        teacher = load_teacher(t_full, device)

        print("Evaluating teacher AccTop1 (GraphPrimaryHead pipeline) ...")
        t_acc, t_correct, t_total = compute_acctop1_teacher(teacher, dl_val, device)
        print(f"  Teacher AccTop1 (graph) = {t_acc:.4f}  ({t_correct}/{t_total})")

        print("\nEvaluating teacher AccTop1 (argmax P_logits with T_star) ...")
        t_acc2, t_correct2, t_total2 = compute_acctop1_teacher_grid(teacher, dl_val, device)
        print(f"  Teacher AccTop1 (grid)  = {t_acc2:.4f}  ({t_correct2}/{t_total2})")

        print(f"\n{'='*60}")
        print(f"  Student AccTop1 (grid):        {acc:.4f}")
        print(f"  Teacher AccTop1 (graph head):  {t_acc:.4f}")
        print(f"  Teacher AccTop1 (grid):        {t_acc2:.4f}")
        print(f"  Student vs Teacher(graph): gap={acc - t_acc:+.4f}  retention={acc/t_acc:.1%}")
        print(f"  Student vs Teacher(grid):  gap={acc - t_acc2:+.4f}  retention={acc/t_acc2:.1%}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
