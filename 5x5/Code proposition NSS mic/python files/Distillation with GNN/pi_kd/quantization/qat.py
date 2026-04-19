"""
Quantization-Aware Training (QAT) fine-tuning for FPGA deployment.

Two-phase schedule:

  Phase 1 — Calibration (n_calibration epochs):
      observers ON, fake_quant OFF.
      Observers see float32 activations and set initial INT8 ranges.
      Fake_quant must stay OFF here — uninitialized ranges on epoch 0 would
      inject pure noise into gradients, instantly destroying pretrained weights.

  Phase 2 — Adaptation (remaining epochs):
      observers ON, fake_quant ON.
      Model adjusts weights to work under INT8 constraints while MinMaxObserver
      tracks activation extremes.

Observer: MinMaxObserver (global min/max, monotonically expanding).
  Root cause of previous collapse at epoch 5: the default qnnpack qconfig uses
  FusedMovingAvgObsFakeQuantize (EMA, averaging_constant=0.01). After ~11,000
  batches (3 calibration + 2 adaptation epochs), the EMA-tracked min/max drifts
  enough that the integer zero_point must jump by 1 — a discontinuous grid
  shift. Every quantized primary heatmap pixel remaps to wrong INT8 bins,
  correct pixel drops from rank #1 to rank #15+, AccTop1 crashes 0.99→0.008.
  Fix: use averaging_constant=0.001 (10x slower EMA). Calibration still
  converges in 3 epochs; drift over 12 adaptation epochs is 10x smaller —
  zero_point jump threshold crossed at ~50 epochs, well beyond our 15-epoch run.

Backend: qnnpack (per-tensor weights) — simpler hardware mapping for hls4ml/FPGA.

Note on double fake_quant: PyTorch does not support Conv-BN-ReLU6 fusion
(only Conv-BN-ReLU). After prepare_qat, each DepthwiseSepBlock has two fake
quantizers (one after Conv-BN, one after ReLU6). This is suboptimal but
harmless with a stable observer: the pre-ReLU6 fake quantizer clips negatives
to [zero_point], which ReLU6 then clips to 0 anyway.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization.observer import MovingAverageMinMaxObserver
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


def prepare_for_qat(student, ckpt_path, device="cpu"):
    """
    Load student from checkpoint, fuse BN into Conv, and insert fake quantizers.

    Fake_quant is left DISABLED — calibration in qat_finetune() enables it
    only after observers have collected representative activation statistics.

    qconfig: qnnpack with MinMaxObserver (replaces default EMA observer).
      - reduce_range=False: full 8-bit (256 levels). reduce_range=True halves
        precision to 127 levels → 2x more quantization noise → model collapse.
      - per-tensor weights: simpler than per-channel, confirmed to work.
      - MinMaxObserver: monotonically expanding range, zero_point never jumps
        discontinuously (see module docstring for why this matters).

    Returns:
        (model, cfg) — model ready for calibration + QAT.
    """
    from ..models.student import StudentNet

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    widths = tuple(cfg.get("student", {}).get("widths", [32, 64, 64]))

    model = StudentNet(in_ch=3, widths=widths)
    model.load_state_dict(ckpt["student"])
    model.to(device)
    model.eval()
    model.fuse_bn()
    model.train()

    torch.backends.quantized.engine = "qnnpack"

    # EMA observer with averaging_constant=0.001 (10x slower than qnnpack default 0.01).
    #
    # Why not MinMaxObserver: sensitive to outlier batches — one extreme batch in 6600
    # calibration steps permanently inflates max_val, scale becomes too coarse, model
    # collapses at the first fake_quant epoch (confirmed: ep3 loss 0.0002→0.0582, 291x).
    #
    # Why not default EMA (0.01): drift accumulates over fake_quant epochs — after
    # 2 adaptation epochs the EMA-tracked max_val crosses a round() boundary,
    # zero_point jumps by 1 integer, remapping all heatmap pixels to wrong INT8 bins →
    # AccTop1 0.99→0.008 (confirmed in every run with averaging_constant=0.01).
    #
    # averaging_constant=0.001: EMA converges in ~3 calibration epochs
    # ((1-0.001)^6600 ≈ 0), but drifts 10x more slowly during adaptation.
    # Zero_point jump threshold crossed at ~50 adaptation epochs (vs 2 with 0.01).
    # With only 12 adaptation epochs (n_adapt=12), no jump occurs.
    _slow_act = quant.FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver.with_args(averaging_constant=0.001),
        quant_min=0, quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    )
    _slow_wt = quant.FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver.with_args(averaging_constant=0.001),
        quant_min=-128, quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    )
    model.qconfig = quant.QConfig(activation=_slow_act, weight=_slow_wt)
    quant.prepare_qat(model, inplace=True)

    model.apply(quant.disable_fake_quant)
    model.apply(quant.enable_observer)

    print("  QAT: EMA observer (averaging_constant=0.001), reduce_range=False, per-tensor | "
          "fake_quant disabled until calibration complete")
    return model, cfg


def evaluate_qat(model, dl_val, device="cpu"):
    """
    Evaluate QAT model.

    Composite metric = AUPRC_Primary (all events) — matches PI-KD objective.
    Also reports AccTop1, AUPRC_Primary_ICS, Primary_F1.
    """
    model.eval()
    all_P_prob, all_P_true = [], []
    all_S_prob, all_S_true = [], []
    ics_P_prob, ics_P_true = [], []
    correct_top1, total_ics = 0, 0

    with torch.no_grad():
        for x, y in dl_val:
            x = x.to(device)
            out = model(x)

            P_prob = torch.sigmoid(out["P_logits"]).squeeze(1).cpu()
            S_prob = torch.sigmoid(out["S_logits"]).squeeze(1).cpu()
            P_star = y["P_star"].cpu()
            S_star = y["S_star"].cpu()
            is_ics = y["is_ics"].cpu()

            all_P_prob.append(P_prob.reshape(-1))
            all_P_true.append(P_star.reshape(-1))
            all_S_prob.append(S_prob.reshape(-1))
            all_S_true.append(S_star.reshape(-1))

            for b in range(x.shape[0]):
                if is_ics[b]:
                    total_ics += 1
                    pred_idx = P_prob[b].flatten().argmax().item()
                    true_idx = P_star[b].flatten().argmax().item()
                    if pred_idx == true_idx:
                        correct_top1 += 1
                    ics_P_prob.append(P_prob[b].flatten())
                    ics_P_true.append(P_star[b].flatten())

    metrics = {}
    metrics["AccTop1"] = correct_top1 / max(total_ics, 1)

    P_prob_np = torch.cat(all_P_prob).numpy()
    P_true_np = torch.cat(all_P_true).numpy()
    if P_true_np.sum() > 0:
        metrics["AUPRC_Primary"] = average_precision_score(P_true_np, P_prob_np)
        prec, rec, _ = precision_recall_curve(P_true_np, P_prob_np)
        f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
        metrics["Primary_F1"] = float(f1s.max())

    S_prob_np = torch.cat(all_S_prob).numpy()
    S_true_np = torch.cat(all_S_true).numpy()
    if S_true_np.sum() > 0:
        metrics["AUPRC_Scatter"] = average_precision_score(S_true_np, S_prob_np)

    if ics_P_prob:
        ics_prob_np = torch.cat(ics_P_prob).numpy()
        ics_true_np = torch.cat(ics_P_true).numpy()
        if ics_true_np.sum() > 0:
            metrics["AUPRC_Primary_ICS"] = average_precision_score(ics_true_np, ics_prob_np)
            prec, rec, _ = precision_recall_curve(ics_true_np, ics_prob_np)
            f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-8)
            metrics["Primary_F1_ICS"] = float(f1s.max())

    return metrics


def _qat_val_score(metrics: dict) -> float:
    """Composite score for checkpoint selection: AUPRC_Primary_ICS (ICS events only).

    Class imbalance (~92% non-ICS) makes all-event AUPRC uninformative.
    The operational pipeline first classifies ICS events, then locates the
    primary pixel within them — so ICS-event-only AUPRC is the right target.
    """
    return float(metrics.get("AUPRC_Primary_ICS", 0))


def qat_finetune(model, dl_train, dl_val, cfg, device="cpu",
                 epochs=15, ckpt_dir="./checkpoints_qat"):
    """
    QAT fine-tuning loop with three-phase observer/fake_quant schedule.
    See module docstring for phase details.
    """
    from ..losses.focal_loss import FocalLoss

    os.makedirs(ckpt_dir, exist_ok=True)

    qat_cfg = cfg.get("qat", {})
    lr_base = cfg.get("train", {}).get("lr", 3e-4)
    lr_qat = lr_base * qat_cfg.get("lr_factor", 0.01)
    eta_min = float(qat_cfg.get("cosine_eta_min", 1e-7))
    n_calibration = int(qat_cfg.get("n_calibration", 3))
    n_adapt = int(qat_cfg.get("n_adapt", 2))
    freeze_epoch = n_calibration + n_adapt

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr_qat, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=eta_min,
    )

    focal_S = FocalLoss(gamma=2.0)
    focal_P = FocalLoss(gamma=2.0)

    w_scatter = cfg.get("loss_weights", {}).get("scatter", 0.15)
    w_primary = cfg.get("loss_weights", {}).get("primary", 1.0)

    best_score = -1.0
    best_metrics = {}
    best_epoch = -1
    prev_loss = None
    no_improve = 0
    qat_patience = int(qat_cfg.get("patience", 3))

    print(f"\n{'='*70}")
    print(f"QAT Fine-tuning: {epochs} epochs, lr={lr_qat:.1e}")
    print(f"  Phase 1 calibration: epochs 0–{n_calibration-1}  "
          f"(observers ON, fake_quant OFF)")
    print(f"  Phase 2 adaptation:  epochs {n_calibration}–{epochs-1}  "
          f"(MinMaxObserver ON, fake_quant ON)")
    print(f"  Task weights: scatter={w_scatter}, primary={w_primary}")
    print(f"  Checkpoint dir: {ckpt_dir}")
    print(f"{'='*70}\n")

    # Pre-QAT baseline: fake_quant disabled → pure float32 performance.
    # This is the target to preserve after quantization.
    pre_metrics = evaluate_qat(model, dl_val, device)
    print(f"PRE-QAT  | AccTop1={pre_metrics.get('AccTop1', 0):.4f} "
          f"AUPRC_Primary_ICS={pre_metrics.get('AUPRC_Primary_ICS', 0):.4f} "
          f"P_ICS_F1={pre_metrics.get('Primary_F1_ICS', 0):.4f} "
          f"AUPRC_Primary_all={pre_metrics.get('AUPRC_Primary', 0):.4f}\n")

    try:
        _tqdm_off = not os.isatty(sys.stdout.fileno())
    except Exception:
        _tqdm_off = True

    for epoch in range(epochs):
        if epoch == 0:
            model.apply(quant.enable_observer)
            model.apply(quant.disable_fake_quant)
        elif epoch == n_calibration:
            # Phase 2: enable fake_quant, keep observers ON for adaptation.
            # The model needs n_adapt epochs to adjust weights to INT8 noise
            # while observers re-calibrate on quantized activations.
            model.apply(quant.enable_fake_quant)
            print(f"\n  [Phase 2] FakeQuant ENABLED at epoch {epoch} "
                  f"(observers still ON for {n_adapt} adaptation epoch(s))\n")
        elif epoch == freeze_epoch:
            # Phase 3: freeze observers — ranges are now calibrated on quantized
            # activations. Lock them to prevent EMA drift from causing instability.
            model.apply(quant.disable_observer)
            print(f"\n  [Phase 3] Observers FROZEN at epoch {epoch}\n")

        model.train()
        losses = []

        pbar = tqdm(dl_train, desc=f"QAT Ep {epoch:3d}", leave=False, disable=_tqdm_off)
        for x, y in pbar:
            x = x.to(device)
            y_dev = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in y.items()}

            optimizer.zero_grad()
            out = model(x)

            l_s = focal_S(out["S_logits"], y_dev["S_star"].unsqueeze(1))
            l_p = focal_P(out["P_logits"], y_dev["P_star"].unsqueeze(1))
            loss = w_scatter * l_s + w_primary * l_p

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()

        avg_loss = np.mean(losses)

        # Skip full eval during calibration: fake_quant is off → float32 baseline,
        # not meaningful to compare against later QAT epochs.
        if epoch < n_calibration:
            print(f"QAT Ep {epoch:3d} [calibration] | loss={avg_loss:.4f}")
            continue

        # Loss-spike guard: compare consecutive ADAPTATION epochs only (not vs calibration).
        # Fires if loss jumps >20x after the model was stable (prev_loss < 0.005).
        # Calibration loss is excluded from comparison — adaptation loss starts higher
        # (fake_quant noise) and is not comparable to float32 calibration loss.
        if epoch > n_calibration and prev_loss is not None and prev_loss < 0.005 and avg_loss > prev_loss * 20:
            print(f"\n  [Loss spike] loss {prev_loss:.4f} → {avg_loss:.4f} "
                  f"({avg_loss/prev_loss:.0f}x jump) — quantization collapse detected, stopping.")
            break

        val_metrics = evaluate_qat(model, dl_val, device)
        score = _qat_val_score(val_metrics)

        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch
            best_metrics = val_metrics.copy()
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "student": model.state_dict(),
                "cfg": cfg,
                "metrics": val_metrics,
            }, os.path.join(ckpt_dir, "best_qat.pt"))
        else:
            no_improve += 1

        print(
            f"QAT Ep {epoch:3d} | loss={avg_loss:.4f} | "
            f"AccTop1={val_metrics.get('AccTop1', 0):.4f} "
            f"AUPRC_Primary_ICS={val_metrics.get('AUPRC_Primary_ICS', 0):.4f} "
            f"P_ICS_F1={val_metrics.get('Primary_F1_ICS', 0):.4f} "
            f"AUPRC_Primary_all={val_metrics.get('AUPRC_Primary', 0):.4f} "
            f"score={score:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"{'*' if is_best else f'(no_improve={no_improve}/{qat_patience})'}"
        )
        prev_loss = avg_loss

        if no_improve >= qat_patience:
            print(f"\n  [Early stop] No improvement for {qat_patience} epochs, stopping.")
            break

    print(f"\nQAT complete. Best AUPRC_Primary_ICS: {best_score:.4f} (epoch {best_epoch})")

    best_path = os.path.join(ckpt_dir, "best_qat.pt")
    if best_epoch >= 0 and os.path.isfile(best_path):
        sd = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["student"])
        print(f"  Restored best weights from epoch {best_epoch}.")

    post_metrics = evaluate_qat(model, dl_val, device)
    print(f"\nPOST-QAT | AccTop1={post_metrics.get('AccTop1', 0):.4f} "
          f"AUPRC_Primary_ICS={post_metrics.get('AUPRC_Primary_ICS', 0):.4f} "
          f"P_ICS_F1={post_metrics.get('Primary_F1_ICS', 0):.4f} "
          f"AUPRC_Primary_all={post_metrics.get('AUPRC_Primary', 0):.4f}")

    delta_top1 = post_metrics.get("AccTop1", 0) - pre_metrics.get("AccTop1", 0)
    delta_auprc_ics = post_metrics.get("AUPRC_Primary_ICS", 0) - pre_metrics.get("AUPRC_Primary_ICS", 0)
    print(f"  ΔAccTop1={delta_top1:+.4f}  ΔAUPRC_Primary_ICS={delta_auprc_ics:+.4f}")
    if abs(delta_top1) > 0.01:
        print(f"  WARNING: AccTop1 shifted >{abs(delta_top1)*100:.1f}% — check QAT quality")

    return model, best_metrics


def convert_and_save(model, save_path="student_quantized.pt"):
    """
    Convert QAT model to fully quantized INT8 model and save state_dict.

    qnnpack/fbgemm backends require CPU for convert().
    """
    model.eval()
    if any(p.is_cuda for p in model.parameters()):
        model = model.cpu()
    model_q = quant.convert(model, inplace=False)
    torch.save(model_q.state_dict(), save_path)
    print(f"Quantized model saved to {save_path}")

    q_size = os.path.getsize(save_path) / 1e6
    try:
        float_size = sum(
            p.numel() * p.element_size() for p in model_q.parameters()
        ) / 1e6
    except Exception:
        float_size = 0.0
    print(f"  File size: {q_size:.2f} MB  |  float param reference: {float_size:.2f} MB")
    return model_q
