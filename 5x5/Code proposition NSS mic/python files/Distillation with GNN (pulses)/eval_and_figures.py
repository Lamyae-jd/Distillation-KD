"""
eval_and_figures.py — Evaluate the pulse-trained PhysFormer teacher on the ICS pool
and produce convergence + prediction figures.

Outputs (under ./figures/):
  - convergence.png           training loss + key val metrics per epoch
  - pr_curve_primary_ics.png  precision-recall on the ICS pool
  - confusion_top1_ics.png    top-1 accuracy summary on ICS pool
  - examples_predictions.png  N pulses: input E, GT primary, predicted P heatmap

Prints AccTop1, AUPRC_Primary_ICS, Primary_F1_ICS on the ICS pool (parity with
../Distillation with GNN/eval_teacher_ics.py).

Run from the pulse workspace:
    python3 eval_and_figures.py
"""
import os, sys, json, math
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pi_kd.data.datamodule import load_data
from pi_kd.models.teacher import load_teacher, teacher_inference

CONFIG       = os.path.join(SCRIPT_DIR, "pi_kd", "configs", "train_config.yaml")
TEACHER_CKPT = os.path.join(SCRIPT_DIR, "checkpoints", "best.pt")
METRICS_LOG  = os.path.join(SCRIPT_DIR, "logs", "train_metrics.jsonl")
FIG_DIR      = os.path.join(SCRIPT_DIR, "figures")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(FIG_DIR, exist_ok=True)


def best_f1(probs, labels):
    p, r, t = precision_recall_curve(labels, probs)
    f1 = 2 * p * r / np.maximum(p + r, 1e-8)
    i = int(np.argmax(f1))
    return float(f1[i]), float(t[i] if i < len(t) else 0.5), p, r


def run_inference(teacher, dl_val):
    P_prob_batches   = []   # (B, 5, 5) sigmoid outputs
    P_star_batches   = []   # (B, 5, 5) ground truth primary map
    is_ics_batches   = []   # (B,) bool
    E_batches        = []   # (B, 5, 5) input energy channel
    pulse_id_batches = []

    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(DEVICE)
            T_star = y["T_star"].unsqueeze(1).to(DEVICE)
            out = teacher_inference(teacher, x, T_star=T_star)
            P_prob = torch.sigmoid(out["P_logits"]).squeeze(1).cpu().numpy()   # (B,5,5)

            P_prob_batches.append(P_prob)
            P_star_batches.append(y["P_star"].cpu().numpy())
            is_ics_batches.append(y["is_ics"].cpu().numpy().astype(bool))
            E_batches.append(x[:, 0].cpu().numpy())
            pulse_id_batches.append(y["pulse_id"].cpu().numpy())

    return (np.concatenate(P_prob_batches, axis=0),
            np.concatenate(P_star_batches, axis=0),
            np.concatenate(is_ics_batches, axis=0),
            np.concatenate(E_batches,      axis=0),
            np.concatenate(pulse_id_batches, axis=0))


def compute_ics_metrics(P_prob, P_star, is_ics):
    P_prob_ics = P_prob[is_ics]
    P_star_ics = P_star[is_ics]

    P_flat   = P_prob_ics.reshape(len(P_prob_ics), -1)
    Pstar_fl = P_star_ics.reshape(len(P_star_ics), -1)

    has_primary = Pstar_fl.sum(axis=1) > 0
    pred_idx   = P_flat.argmax(axis=1)
    gt_at_pred = Pstar_fl[np.arange(len(pred_idx)), pred_idx]
    top1_correct = int((gt_at_pred[has_primary] > 0).sum())
    top1_total   = int(has_primary.sum())
    acc_top1     = top1_correct / max(top1_total, 1)

    labels_flat = Pstar_fl.reshape(-1)
    probs_flat  = P_flat.reshape(-1)
    auprc_ics   = average_precision_score(labels_flat, probs_flat)
    f1_ics, thr_ics, prec, rec = best_f1(probs_flat, labels_flat)

    return dict(
        acc_top1=acc_top1, top1_correct=top1_correct, top1_total=top1_total,
        auprc=auprc_ics, best_f1=f1_ics, best_thr=thr_ics,
        pr_prec=prec, pr_rec=rec,
        n_ics=int(is_ics.sum()), n_total=len(is_ics),
    )


def plot_convergence(path=METRICS_LOG, out=None):
    if not os.path.exists(path):
        print(f"[skip] no metrics log at {path}")
        return
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print(f"[skip] empty metrics log")
        return
    ep = [r["epoch"] for r in rows]
    loss = [r.get("train_loss") for r in rows]
    keys = ["Primary_AccTop1", "Primary_F1", "AUPRC_PrimaryCandidates",
            "AUPRC_ICS", "AUPRC_Scatter"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ep, loss, "o-", color="tab:red", label="train_loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Training loss"); axes[0].grid(alpha=0.3)
    axes[0].legend()

    for k in keys:
        vals = [r.get(k) for r in rows]
        if any(v is not None for v in vals):
            axes[1].plot(ep, vals, "o-", label=k)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Metric")
    axes[1].set_title("Validation metrics")
    axes[1].set_ylim(0, 1.02); axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)

    fig.suptitle("Pulse teacher — convergence", fontsize=12)
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "convergence.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


def plot_pr_ics(m, out=None):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.plot(m["pr_rec"], m["pr_prec"], color="tab:blue",
            label=f"AUPRC={m['auprc']:.3f}\nBest F1={m['best_f1']:.3f} @ thr={m['best_thr']:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"Primary head — ICS pool (n={m['n_ics']})")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "pr_curve_primary_ics.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


def plot_top1_summary(m, out=None):
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    ok  = m["top1_correct"]
    bad = m["top1_total"] - ok
    ax.bar(["Correct top-1", "Wrong top-1"], [ok, bad],
           color=["tab:green", "tab:red"])
    ax.set_ylabel("# ICS pulses with a primary")
    ax.set_title(f"AccTop1 = {m['acc_top1']:.4f}  ({ok}/{m['top1_total']})")
    for i, v in enumerate([ok, bad]):
        ax.text(i, v, f"{v}", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "confusion_top1_ics.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


def plot_examples(P_prob, P_star, is_ics, E, pulse_ids, n=8, out=None):
    ics_idx = np.where(is_ics)[0]
    if len(ics_idx) == 0:
        print("[skip] no ICS pulses in val set")
        return

    P_flat = P_prob[ics_idx].reshape(len(ics_idx), -1)
    Ps_fl  = P_star[ics_idx].reshape(len(ics_idx), -1)
    has_p  = Ps_fl.sum(axis=1) > 0
    ics_with_prim = ics_idx[has_p]
    P_flat = P_flat[has_p]; Ps_fl = Ps_fl[has_p]

    pred = P_flat.argmax(axis=1)
    gt_hit = Ps_fl[np.arange(len(pred)), pred] > 0

    n_ok = int(gt_hit.sum()); n_bad = int((~gt_hit).sum())
    take_ok  = min(n // 2, n_ok)
    take_bad = min(n - take_ok, n_bad)
    take_ok  = min(n - take_bad, n_ok)  # rebalance if bad short

    ok_pool  = ics_with_prim[gt_hit]
    bad_pool = ics_with_prim[~gt_hit]

    rng = np.random.default_rng(0)
    ok_sel  = rng.choice(ok_pool,  size=take_ok,  replace=False) if take_ok  else np.array([], dtype=int)
    bad_sel = rng.choice(bad_pool, size=take_bad, replace=False) if take_bad else np.array([], dtype=int)
    sel = np.concatenate([ok_sel, bad_sel])
    labels = ["OK"] * len(ok_sel) + ["MISS"] * len(bad_sel)

    n_show = len(sel)
    if n_show == 0:
        print("[skip] no ICS examples with a primary label")
        return

    fig, axes = plt.subplots(3, n_show, figsize=(2.2 * n_show, 6.6))
    if n_show == 1:
        axes = axes[:, None]

    for i, (idx, lab) in enumerate(zip(sel, labels)):
        E_map    = E[idx]
        Pstar    = P_star[idx]
        Pprob    = P_prob[idx]
        pred_idx = int(Pprob.argmax())
        pr, pc   = pred_idx // 5, pred_idx % 5

        gt_pixels = np.argwhere(Pstar > 0)

        for ax, arr, title, cmap in zip(
                axes[:, i],
                [E_map, Pstar, Pprob],
                [f"Input E\npulse={int(pulse_ids[idx])}", "Ground truth P*", "Predicted P"],
                ["viridis", "Greens", "magma"]):
            im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for r, c in gt_pixels:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1,
                                            fill=False, edgecolor="lime", lw=1.5))
            if title.startswith("Predicted"):
                color = "cyan" if lab == "OK" else "red"
                ax.add_patch(plt.Rectangle((pc - 0.5, pr - 0.5), 1, 1,
                                            fill=False, edgecolor=color, lw=1.8))
            if i == 0:
                ax.set_ylabel(title.split("\n")[0], fontsize=9)
            if title.startswith("Input"):
                ax.set_title(f"{title}\n[{lab}]", fontsize=8)
        axes[1, i].set_title("GT (green)", fontsize=8)
        axes[2, i].set_title("Pred argmax", fontsize=8)

    fig.suptitle("ICS pulses — GT primary (green box) vs top-1 prediction "
                 "(cyan=hit, red=miss)", fontsize=11)
    fig.tight_layout()
    out = out or os.path.join(FIG_DIR, "examples_predictions.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  saved {out}")


def main():
    print(f"Device: {DEVICE}")
    if not os.path.exists(TEACHER_CKPT):
        print(f"[error] no checkpoint at {TEACHER_CKPT} — run train.py first.")
        return 1

    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    print("Loading val set...")
    _, dl_val, _ = load_data(cfg, verbose=False)

    print(f"Loading teacher from {TEACHER_CKPT} ...")
    teacher = load_teacher(TEACHER_CKPT, DEVICE)

    print("Running inference on val set...")
    P_prob, P_star, is_ics, E, pulse_ids = run_inference(teacher, dl_val)
    print(f"  {len(P_prob):,} val pulses | {int(is_ics.sum()):,} ICS")

    m = compute_ics_metrics(P_prob, P_star, is_ics)
    print("\n=== TEACHER metrics — ICS pool ===")
    print(f"  AccTop1               : {m['acc_top1']:.4f}  ({m['top1_correct']}/{m['top1_total']})")
    print(f"  AUPRC_Primary_ICS     : {m['auprc']:.4f}")
    print(f"  Primary_F1_ICS        : {m['best_f1']:.4f}  thr={m['best_thr']:.4f}")
    print(f"  ICS pool size (val)   : {m['n_ics']} / {m['n_total']}")

    print("\nGenerating figures...")
    plot_convergence()
    plot_pr_ics(m)
    plot_top1_summary(m)
    plot_examples(P_prob, P_star, is_ics, E, pulse_ids, n=8)

    summary = {
        "AccTop1_ICS":           m["acc_top1"],
        "AUPRC_Primary_ICS":     m["auprc"],
        "Primary_F1_ICS":        m["best_f1"],
        "Primary_thr_ICS":       m["best_thr"],
        "n_ics_val":             m["n_ics"],
        "n_total_val":           m["n_total"],
        "top1_correct":          m["top1_correct"],
        "top1_total":            m["top1_total"],
    }
    with open(os.path.join(FIG_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {os.path.join(FIG_DIR, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
