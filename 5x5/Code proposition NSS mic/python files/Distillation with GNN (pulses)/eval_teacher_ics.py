"""Evaluate teacher on ICS pool — same metrics as trainer.validate()."""
import os, sys, yaml, numpy as np, torch
from sklearn.metrics import average_precision_score, precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pi_kd.data.datamodule import load_data
from pi_kd.models.teacher import load_teacher, teacher_inference

CONFIG = os.path.join(SCRIPT_DIR, "pi_kd", "configs", "train_config.yaml")
TEACHER_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "best.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def best_f1(probs, labels):
    p, r, t = precision_recall_curve(labels, probs)
    f1 = 2 * p * r / np.maximum(p + r, 1e-8)
    i = int(np.argmax(f1))
    return float(f1[i]), float(t[i] if i < len(t) else 0.5)


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)
    print(f"Device: {DEVICE}")
    _, dl_val, _ = load_data(cfg, verbose=False)
    teacher = load_teacher(TEACHER_CKPT, DEVICE)

    all_P_prob, all_P_true = [], []
    ics_P_prob, ics_P_true = [], []
    top1_correct, top1_total = 0, 0

    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(DEVICE)
            T_star = y["T_star"].unsqueeze(1).to(DEVICE)
            out = teacher_inference(teacher, x, T_star=T_star)
            P_prob = torch.sigmoid(out["P_logits"]).squeeze(1).cpu()
            is_ics = y["is_ics"].cpu()

            P_flat = P_prob.view(P_prob.shape[0], -1)
            P_true_flat = y["P_star"].cpu().view(P_prob.shape[0], -1)
            pred_idx = P_flat.argmax(dim=1)
            has_primary = P_true_flat.sum(dim=1) > 0
            gt_at_pred = P_true_flat[torch.arange(len(pred_idx)), pred_idx]
            top1_correct += int((gt_at_pred[has_primary] > 0).sum())
            top1_total += int(has_primary.sum())

            all_P_prob.append(P_prob.reshape(-1))
            all_P_true.append(y["P_star"].cpu().reshape(-1))
            for b in range(x.shape[0]):
                if is_ics[b]:
                    ics_P_prob.append(P_prob[b].flatten())
                    ics_P_true.append(y["P_star"][b].cpu().flatten())

    P_all = torch.cat(all_P_prob).numpy()
    P_all_t = torch.cat(all_P_true).numpy()
    P_ics = torch.cat(ics_P_prob).numpy()
    P_ics_t = torch.cat(ics_P_true).numpy()

    auprc_all = average_precision_score(P_all_t, P_all)
    f1_all, thr_all = best_f1(P_all, P_all_t)
    auprc_ics = average_precision_score(P_ics_t, P_ics)
    f1_ics, thr_ics = best_f1(P_ics, P_ics_t)
    acctop1 = top1_correct / max(top1_total, 1)

    print("\n=== TEACHER metrics on val set ===")
    print(f"  AUPRC_Primary (all events) : {auprc_all:.4f}")
    print(f"  Primary_F1   (all events)  : {f1_all:.4f}  thr={thr_all:.4f}")
    print(f"  AUPRC_Primary_ICS (ICS pool): {auprc_ics:.4f}")
    print(f"  Primary_F1_ICS   (ICS pool): {f1_ics:.4f}  thr={thr_ics:.4f}")
    print(f"  AccTop1                    : {acctop1:.4f}")
    print(f"  ICS pool size              : {len(P_ics)//25} events ({P_ics_t.sum():.0f} positives)")


if __name__ == "__main__":
    main()
