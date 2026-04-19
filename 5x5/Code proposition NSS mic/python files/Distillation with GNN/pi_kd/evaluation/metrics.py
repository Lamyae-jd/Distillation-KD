"""
Evaluation metrics for comparing Teacher, Student, Student+KD, Student+PI-KD.
Includes F1, AUPRC, AUROC, and parameter/FLOP counts.
"""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
)


def compute_all_metrics(model, dataloader, device="cpu", threshold=0.5):
    """
    Run inference and compute all metrics.

    Returns:
        dict with scatter, primary, ICS metrics + model stats
    """
    model.eval()
    all_S_prob, all_S_true = [], []
    all_P_prob, all_P_true = [], []
    all_ics_prob, all_ics_true = [], []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            out = model(x)

            S_prob = torch.sigmoid(out["S_logits"]).squeeze(1).cpu().reshape(-1)
            P_prob = torch.sigmoid(out["P_logits"]).squeeze(1).cpu().reshape(-1)
            ics_prob = torch.sigmoid(out["ics_logit"]).cpu()

            all_S_prob.append(S_prob)
            all_S_true.append(y["S_star"].cpu().reshape(-1))
            all_P_prob.append(P_prob)
            all_P_true.append(y["P_star"].cpu().reshape(-1))
            all_ics_prob.append(ics_prob)
            all_ics_true.append(y["is_ics"].float().cpu())

    S_prob = torch.cat(all_S_prob).numpy()
    S_true = torch.cat(all_S_true).numpy()
    P_prob = torch.cat(all_P_prob).numpy()
    P_true = torch.cat(all_P_true).numpy()
    ics_prob = torch.cat(all_ics_prob).numpy()
    ics_true = torch.cat(all_ics_true).numpy()

    metrics = {}

    # --- Scatter ---
    S_pred = (S_prob > threshold).astype(float)
    if S_true.sum() > 0:
        metrics["scatter_auprc"] = average_precision_score(S_true, S_prob)
        metrics["scatter_auroc"] = roc_auc_score(S_true, S_prob)
        tp_s = ((S_pred == 1) & (S_true == 1)).sum()
        fp_s = ((S_pred == 1) & (S_true == 0)).sum()
        fn_s = ((S_pred == 0) & (S_true == 1)).sum()
        metrics["scatter_precision"] = tp_s / max(tp_s + fp_s, 1)
        metrics["scatter_recall"] = tp_s / max(tp_s + fn_s, 1)
        f1_num = 2 * metrics["scatter_precision"] * metrics["scatter_recall"]
        f1_den = metrics["scatter_precision"] + metrics["scatter_recall"]
        metrics["scatter_f1"] = f1_num / max(f1_den, 1e-8)

    # --- Primary ---
    P_pred = (P_prob > threshold).astype(float)
    if P_true.sum() > 0:
        metrics["primary_auprc"] = average_precision_score(P_true, P_prob)
        tp_p = ((P_pred == 1) & (P_true == 1)).sum()
        fp_p = ((P_pred == 1) & (P_true == 0)).sum()
        fn_p = ((P_pred == 0) & (P_true == 1)).sum()
        metrics["primary_precision"] = tp_p / max(tp_p + fp_p, 1)
        metrics["primary_recall"] = tp_p / max(tp_p + fn_p, 1)
        f1_num = 2 * metrics["primary_precision"] * metrics["primary_recall"]
        f1_den = metrics["primary_precision"] + metrics["primary_recall"]
        metrics["primary_f1"] = f1_num / max(f1_den, 1e-8)

    # --- ICS ---
    ics_pred = (ics_prob > threshold).astype(float)
    if ics_true.sum() > 0 and (1 - ics_true).sum() > 0:
        metrics["ics_auprc"] = average_precision_score(ics_true, ics_prob)
        metrics["ics_auroc"] = roc_auc_score(ics_true, ics_prob)
        tp_i = ((ics_pred == 1) & (ics_true == 1)).sum()
        fp_i = ((ics_pred == 1) & (ics_true == 0)).sum()
        fn_i = ((ics_pred == 0) & (ics_true == 1)).sum()
        metrics["ics_precision"] = tp_i / max(tp_i + fp_i, 1)
        metrics["ics_recall"] = tp_i / max(tp_i + fn_i, 1)
        f1_num = 2 * metrics["ics_precision"] * metrics["ics_recall"]
        f1_den = metrics["ics_precision"] + metrics["ics_recall"]
        metrics["ics_f1"] = f1_num / max(f1_den, 1e-8)

    # Composite
    metrics["composite_f1"] = (
        metrics.get("scatter_f1", 0) * 0.4
        + metrics.get("primary_f1", 0) * 0.4
        + metrics.get("ics_f1", 0) * 0.2
    )

    return metrics


def model_stats(model, input_shape=(1, 3, 5, 5)):
    """Count parameters and estimate FLOPs."""
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    stats = {
        "params_total": n_params,
        "params_trainable": n_trainable,
    }

    if hasattr(model, "count_flops"):
        stats["flops"] = model.count_flops(input_shape)

    return stats


def print_comparison(results: dict):
    """
    Pretty-print a comparison table.

    Args:
        results: dict of {config_name: metrics_dict}
    """
    configs = list(results.keys())
    all_keys = set()
    for m in results.values():
        all_keys |= set(m.keys())
    keys = sorted(all_keys)

    # Header
    header = f"{'Metric':<25s}"
    for c in configs:
        header += f" {c:>18s}"
    print(header)
    print("-" * len(header))

    for k in keys:
        row = f"{k:<25s}"
        for c in configs:
            v = results[c].get(k, None)
            if v is None:
                row += f" {'--':>18s}"
            elif isinstance(v, float):
                row += f" {v:>18.4f}"
            else:
                row += f" {v:>18,}"
        print(row)
