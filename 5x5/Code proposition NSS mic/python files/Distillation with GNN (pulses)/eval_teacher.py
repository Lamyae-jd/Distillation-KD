#!/usr/bin/env python3
"""
Quick diagnostic: evaluate teacher and student (from checkpoint) on the val set.
Prints F1, AUPRC, and optimal threshold for each head.

Run from the 'Distillation with GNN' directory:
    python eval_teacher.py
"""

import os, sys
import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pi_kd.data.datamodule import load_data
from pi_kd.models.teacher import load_teacher
from pi_kd.models.student import StudentNet

CONFIG = os.path.join(SCRIPT_DIR, "pi_kd", "configs", "train_config.yaml")
TEACHER_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "best.pt")
STUDENT_CKPT = os.path.join(SCRIPT_DIR, "checkpoints_pikd", "best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, dl_val, device, is_student=False):
    model.eval()
    all_S_prob, all_S_true = [], []
    all_P_prob, all_P_true = [], []
    all_ics_prob, all_ics_true = [], []

    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device)
            out = model(x)
            S_prob = torch.sigmoid(out["S_logits"]).squeeze(1).cpu().numpy().reshape(-1)
            P_prob = torch.sigmoid(out["P_logits"]).squeeze(1).cpu().numpy().reshape(-1)
            ics_prob = torch.sigmoid(out["ics_logit"]).cpu().numpy()

            all_S_prob.append(S_prob)
            all_S_true.append(y["S_star"].numpy().reshape(-1))
            all_P_prob.append(P_prob)
            all_P_true.append(y["P_star"].numpy().reshape(-1))
            all_ics_prob.append(ics_prob)
            all_ics_true.append(y["is_ics"].float().numpy())

    S_prob = np.concatenate(all_S_prob)
    S_true = np.concatenate(all_S_true)
    P_prob = np.concatenate(all_P_prob)
    P_true = np.concatenate(all_P_true)
    ics_prob = np.concatenate(all_ics_prob)
    ics_true = np.concatenate(all_ics_true)

    def best_f1(probs, labels):
        precision, recall, thresholds = precision_recall_curve(labels, probs)
        f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
        best_idx = np.argmax(f1s)
        return f1s[best_idx], thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    results = {}
    for name, prob, true in [("Scatter", S_prob, S_true),
                              ("Primary", P_prob, P_true),
                              ("ICS",     ics_prob, ics_true)]:
        if true.sum() == 0:
            continue
        auprc = average_precision_score(true, prob)
        f1_05 = f1_score(true, (prob > 0.5).astype(int), zero_division=0)
        best_f1_val, best_thr = best_f1(prob, true)
        pos_frac = true.mean()
        results[name] = dict(auprc=auprc, f1_05=f1_05,
                             best_f1=best_f1_val, best_thr=best_thr,
                             pos_frac=pos_frac)
    return results


def print_results(name, results):
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"{'─'*60}")
    print(f"  {'Head':<10} {'pos%':>6} {'AUPRC':>7} {'F1@0.5':>8} {'BestF1':>8} {'BestThr':>9}")
    for head, m in results.items():
        print(f"  {head:<10} {m['pos_frac']*100:>5.1f}% "
              f"{m['auprc']:>7.4f} {m['f1_05']:>8.4f} "
              f"{m['best_f1']:>8.4f} {m['best_thr']:>9.4f}")


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    print(f"Device: {DEVICE}")
    print("Loading validation data...")
    _, dl_val, _ = load_data(cfg, verbose=False)

    # --- Teacher ---
    if os.path.exists(TEACHER_CKPT):
        print(f"\nEvaluating teacher ({TEACHER_CKPT})...")
        teacher = load_teacher(TEACHER_CKPT, DEVICE)
        t_res = evaluate(teacher, dl_val, DEVICE)
        print_results("TEACHER (PhysFormer)", t_res)
    else:
        print(f"\nTeacher checkpoint not found: {TEACHER_CKPT}")

    # --- Student ---
    if os.path.exists(STUDENT_CKPT):
        print(f"\nEvaluating student ({STUDENT_CKPT})...")
        ckpt = torch.load(STUDENT_CKPT, map_location=DEVICE, weights_only=False)
        s_cfg = cfg.get("student", {})
        student = StudentNet(
            in_ch=s_cfg.get("in_ch", 3),
            widths=tuple(s_cfg.get("widths", [32, 64, 64])),
        ).to(DEVICE)
        student.load_state_dict(ckpt["student"])
        s_res = evaluate(student, dl_val, DEVICE)
        print_results("STUDENT (best checkpoint)", s_res)
    else:
        print(f"\nStudent checkpoint not found: {STUDENT_CKPT}")

    # --- Gap analysis ---
    if 'teacher' in dir() and os.path.exists(STUDENT_CKPT):
        print(f"\n{'─'*60}")
        print("  Gap analysis (student / teacher)")
        print(f"{'─'*60}")
        for head in t_res:
            if head in s_res:
                gap_auprc = s_res[head]['auprc'] / max(t_res[head]['auprc'], 1e-8)
                gap_bf1   = s_res[head]['best_f1'] / max(t_res[head]['best_f1'], 1e-8)
                print(f"  {head:<10}  AUPRC: {gap_auprc*100:.1f}% of teacher  |  BestF1: {gap_bf1*100:.1f}% of teacher")


if __name__ == "__main__":
    main()
