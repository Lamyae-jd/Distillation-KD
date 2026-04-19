"""
Ablation study: compare Teacher vs Student (no KD) vs Student+KD vs Student+PI-KD.
"""

import os
import sys
import torch
from torch.utils.data import DataLoader, Subset

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from .metrics import compute_all_metrics, model_stats, print_comparison


def load_student_from_ckpt(ckpt_path, device="cpu"):
    """Load a student model from checkpoint."""
    from ..models.student import StudentNet
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    widths = tuple(cfg.get("student", {}).get("widths", [32, 64, 64]))
    student = StudentNet(in_ch=3, widths=widths)
    student.load_state_dict(ckpt["student"])
    student.to(device).eval()
    return student


def load_teacher_wrapper(ckpt_path, device="cpu"):
    """Load teacher and wrap for evaluation (same output keys as student)."""
    from ..models.teacher import load_teacher
    teacher = load_teacher(ckpt_path, device)

    class TeacherEvalWrapper(torch.nn.Module):
        def __init__(self, t):
            super().__init__()
            self.t = t
        def forward(self, x):
            out = self.t(x)
            return {
                "S_logits": out["S_logits"],
                "P_logits": out["P_logits"],
                "ics_logit": out["ics_logit"],
            }
        def count_flops(self, *a):
            return None

    return TeacherEvalWrapper(teacher)


def run_ablation(
    val_loader,
    teacher_ckpt: str,
    student_no_kd_ckpt: str = None,
    student_kd_ckpt: str = None,
    student_pikd_ckpt: str = None,
    device: str = "cpu",
):
    """
    Run the 4-configuration ablation study.

    Args:
        val_loader: validation DataLoader
        teacher_ckpt: path to teacher best.pt
        student_*_ckpt: paths to student checkpoints (None to skip)
        device: target device
    """
    results = {}

    # Teacher
    print("Evaluating Teacher...")
    teacher = load_teacher_wrapper(teacher_ckpt, device)
    m = compute_all_metrics(teacher, val_loader, device)
    s = model_stats(teacher)
    results["Teacher"] = {**m, **s}

    # Student no KD
    if student_no_kd_ckpt and os.path.exists(student_no_kd_ckpt):
        print("Evaluating Student (no KD)...")
        model = load_student_from_ckpt(student_no_kd_ckpt, device)
        m = compute_all_metrics(model, val_loader, device)
        s = model_stats(model)
        results["Student_noKD"] = {**m, **s}

    # Student + KD
    if student_kd_ckpt and os.path.exists(student_kd_ckpt):
        print("Evaluating Student + KD...")
        model = load_student_from_ckpt(student_kd_ckpt, device)
        m = compute_all_metrics(model, val_loader, device)
        s = model_stats(model)
        results["Student_KD"] = {**m, **s}

    # Student + PI-KD
    if student_pikd_ckpt and os.path.exists(student_pikd_ckpt):
        print("Evaluating Student + PI-KD...")
        model = load_student_from_ckpt(student_pikd_ckpt, device)
        m = compute_all_metrics(model, val_loader, device)
        s = model_stats(model)
        results["Student_PIKD"] = {**m, **s}

    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS")
    print("=" * 80)
    print_comparison(results)

    return results
