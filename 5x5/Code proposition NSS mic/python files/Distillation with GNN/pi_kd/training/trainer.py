"""
PI-KD Trainer: Physics-Informed Knowledge Distillation training loop.

Features:
  - Curriculum-based training (3 phases: task -> +KD -> +physics)
  - Multi-teacher ensemble (averaged logits from multiple checkpoints)
  - Optional feature matching (off by default, enable for ablation)
  - Temperature annealing (T: 2.0 -> 1.2 via cosine schedule)
  - Per-phase LR warmup to avoid disrupting converged weights
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from typing import List, Optional, Union

from ..losses.focal_loss import FocalLoss
from ..losses.distillation_loss import DistillationLoss, FeatureMatchingLoss
from ..losses.physics_loss import PhysicsLoss, PhysicsLossCorreected, PhysicsRankingLoss
from ..training.curriculum import CurriculumScheduler
from ..models.teacher import teacher_inference, multi_teacher_inference


class PIKDTrainer:
    """
    Physics-Informed Knowledge Distillation trainer.

    Supports:
      A. Multi-teacher: pass a list of teachers for ensemble distillation
      B. Feature matching: layer-wise distillation with learned 1x1 projection
      E. Temperature annealing: T decays from T_start to T_end over KD phases

    Args:
        student: StudentNet model
        teacher: frozen PhysFormerWrapper, list of them, or None (ablation)
        optimizer: optimizer for student parameters
        cfg: configuration dict
        device: 'cuda' or 'cpu'
        ckpt_dir: directory for saving checkpoints
    """

    def __init__(self, student, teacher, optimizer, cfg, device="cpu", ckpt_dir="./checkpoints_pikd"):
        self.student = student
        self.device = device

        # A. Multi-teacher support
        if teacher is None:
            self.teachers = None
            self.multi_teacher = False
        elif isinstance(teacher, list):
            self.teachers = teacher
            self.multi_teacher = True
        else:
            self.teachers = [teacher]
            self.multi_teacher = len([teacher]) > 1  # False for single

        self.opt = optimizer
        self.cfg = cfg
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

        # Loss functions
        self.focal_scatter = FocalLoss(gamma=2.0)
        self.focal_primary = FocalLoss(gamma=2.0)
        self.focal_ics = FocalLoss(gamma=2.0)

        kd_cfg = cfg.get("distillation", {})
        self.kd_loss = DistillationLoss(temperature=kd_cfg.get("temperature", 1.5))
        phys_mode = kd_cfg.get("physics_mode", "ranking")
        pitch = cfg.get("data", {}).get("pixel_pitch_mm", (2.0, 2.0))[0]
        if phys_mode == "ranking":
            self.phys_loss = PhysicsRankingLoss(pitch_mm=pitch)
        else:
            self.phys_loss = PhysicsLossCorreected(pitch_mm=pitch)

        # B. Feature matching loss (learned projection student -> teacher space)
        feat_cfg = kd_cfg.get("feature_matching", {})
        student_inter_ch = feat_cfg.get("student_ch", 64)
        teacher_inter_ch = feat_cfg.get("teacher_ch", 64)
        self.feat_loss = FeatureMatchingLoss(student_inter_ch, teacher_inter_ch).to(device)
        self.use_feature_matching = kd_cfg.get("use_feature_matching", True)

        # Pre/post-MHSA KD weighting (Option B: distill CNN-only P1_logits too)
        self.pre_mhsa_weight = kd_cfg.get("pre_mhsa_weight", 0.0)
        self.post_mhsa_weight = kd_cfg.get("post_mhsa_weight", 1.0)

        # Curriculum (with temperature annealing)
        cur_cfg = cfg.get("curriculum", {})
        train_cfg = cfg.get("train", {})
        self.curriculum = CurriculumScheduler(
            total_epochs=train_cfg.get("epochs", 30),
            num_phases=cur_cfg.get("num_phases", 2),
            phase1_frac=cur_cfg.get("phase1_frac", 0.50),
            phase2_frac=cur_cfg.get("phase2_frac", 0.50),
            lambda_kd_max=kd_cfg.get("lambda_kd", 1.0),
            lambda_phys_max=kd_cfg.get("lambda_phys", 0.1),
            lambda_feat_max=kd_cfg.get("lambda_feat", 0.5),
            ramp_epochs=cur_cfg.get("ramp_epochs", 3),
            T_start=kd_cfg.get("T_start", 2.0),
            T_end=kd_cfg.get("T_end", 1.2),
        )

        # Task loss weights
        self.w_scatter = cfg.get("loss_weights", {}).get("scatter", 1.0)
        self.w_primary = cfg.get("loss_weights", {}).get("primary", 2.0)
        self.w_ics = cfg.get("loss_weights", {}).get("ics", 2.0)

        # Mixed precision
        use_amp = (device == "cuda") and cfg.get("mixed_precision", True)
        self.scaler = GradScaler("cuda", enabled=use_amp)
        self.use_amp = use_amp

        # LR warmup + ReduceLROnPlateau
        self.warmup_epochs = train_cfg.get("warmup_epochs", 3)
        self.base_lr = train_cfg.get("lr", 3e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="max", factor=0.5, patience=5,
        )

        self.best_score = -1.0

    def _get_teacher_outputs(self, x: torch.Tensor, T_star: torch.Tensor = None,
                             extract_features: bool = False) -> dict:
        """Get teacher outputs, handling single vs multi-teacher.

        Args:
            T_star: (B, 1, 5, 5) timing map — required for the teacher's
                    PhysicsBiasedMHSA to produce correct attention patterns.
        """
        if self.teachers is None:
            return None
        if len(self.teachers) == 1:
            return teacher_inference(self.teachers[0], x, T_star=T_star,
                                     extract_features=extract_features)
        else:
            return multi_teacher_inference(self.teachers, x, T_star=T_star,
                                           extract_features=extract_features)

    def _task_loss(self, student_out, targets):
        """Compute task losses: focal on scatter, primary, ICS."""
        l_scatter = self.focal_scatter(student_out["S_logits"], targets["S_star"].unsqueeze(1))
        l_primary = self.focal_primary(student_out["P_logits"], targets["P_star"].unsqueeze(1))
        l_ics = self.focal_ics(
            student_out["ics_logit"],
            targets["is_ics"].float(),
        )

        l_task = (
            self.w_scatter * l_scatter
            + self.w_primary * l_primary
            + self.w_ics * l_ics
        )

        return l_task, {
            "scatter": l_scatter.item(),
            "primary": l_primary.item(),
            "ics": l_ics.item(),
        }

    def _apply_warmup_lr(self, epoch, phase_start_epoch: int = 0):
        """Linear LR warmup relative to the start of the current phase."""
        epochs_into_phase = epoch - phase_start_epoch
        if epochs_into_phase < self.warmup_epochs:
            scale = (epochs_into_phase + 1) / self.warmup_epochs
            lr = self.base_lr * (0.1 + 0.9 * scale)
            for pg in self.opt.param_groups:
                pg["lr"] = lr

    def train_one_epoch(self, loader, epoch):
        self.student.train()
        self.feat_loss.train()  # projection layer is trainable
        lambdas = self.curriculum.get_lambdas(epoch)
        phase = self.curriculum.get_phase_name(epoch)
        temperature = lambdas["temperature"]

        losses_acc = {"task": [], "kd_scaled": [], "kd_pre_mhsa": [], "feat_scaled": [], "phys_scaled": [], "total": []}
        bf16_ok = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        amp_dtype = torch.bfloat16 if (self.use_amp and bf16_ok) else torch.float16

        need_kd = lambdas["lambda_kd"] > 0 and self.teachers is not None
        need_feat = lambdas["lambda_feat"] > 0 and self.teachers is not None and self.use_feature_matching
        need_features = need_feat  # extract intermediate features only when needed

        # Disable tqdm when stdout is not a real terminal (e.g. redirected to file with nohup).
        # os.isatty(fileno) is more reliable than sys.stdout.isatty() under nohup.
        try:
            _tqdm_off = not os.isatty(sys.stdout.fileno())
        except Exception:
            _tqdm_off = True
        pbar = tqdm(loader, desc=f"Ep {epoch} [{phase}] T={temperature:.1f}", leave=False,
                    disable=_tqdm_off)
        for x, y in pbar:
            x = x.to(self.device)
            y = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in y.items()}

            self.opt.zero_grad(set_to_none=True)

            with autocast("cuda", enabled=self.use_amp, dtype=amp_dtype):
                student_out = self.student(x)

                # Task loss (always active)
                l_task, task_parts = self._task_loss(student_out, y)
                l_total = l_task

                # Teacher inference (once per batch, shared between KD and feat matching)
                # T_star is the timing map needed by the teacher's PhysicsBiasedMHSA
                teacher_out = None
                if need_kd or need_feat:
                    T_star = y["T_star"].unsqueeze(1).to(self.device)  # (B,5,5) -> (B,1,5,5)
                    teacher_out = self._get_teacher_outputs(
                        x, T_star=T_star, extract_features=need_features
                    )

                # Task magnitude as plain Python float — immune to autocast dtype interference.
                # Using tensor ops (even with .float()) inside autocast can silently lose
                # precision when the result is cast back to float16 by subsequent ops.
                task_mag_f = max(float(l_task.detach()), 1e-7)

                l_kd_scaled = 0.0
                l_kd_pre_mhsa_val = 0.0
                l_kd = torch.tensor(0.0, device=self.device)
                if need_kd and teacher_out is not None:
                    kd_dict = self.kd_loss(
                        student_out, teacher_out, temperature=temperature,
                        pre_mhsa_weight=self.pre_mhsa_weight,
                        post_mhsa_weight=self.post_mhsa_weight,
                    )
                    l_kd = kd_dict["kd_total"]
                    l_kd_pre_mhsa_val = kd_dict["kd_pre_mhsa"].item()
                    kd_mag_f = max(float(l_kd.detach()), 1e-7)
                    adaptive_kd = lambdas["lambda_kd"] * task_mag_f / kd_mag_f
                    l_total = l_total + adaptive_kd * l_kd
                    l_kd_scaled = adaptive_kd * l_kd.item()

                l_feat_scaled = 0.0
                l_feat = torch.tensor(0.0, device=self.device)
                if need_feat and teacher_out is not None and "feat_inter" in teacher_out:
                    l_feat = self.feat_loss(student_out["feat_inter"], teacher_out["feat_inter"])
                    feat_mag_f = max(float(l_feat.detach()), 1e-7)
                    adaptive_feat = lambdas["lambda_feat"] * task_mag_f / feat_mag_f
                    l_total = l_total + adaptive_feat * l_feat
                    l_feat_scaled = adaptive_feat * l_feat.item()

                l_phys_scaled = 0.0
                l_phys = torch.tensor(0.0, device=self.device)
                if lambdas["lambda_phys"] > 0:
                    l_phys = self.phys_loss(student_out, x, y)
                    phys_mag_f = max(float(l_phys.detach()), 1e-7)
                    adaptive_phys = lambdas["lambda_phys"] * task_mag_f / phys_mag_f
                    l_total = l_total + adaptive_phys * l_phys
                    l_phys_scaled = adaptive_phys * l_phys.item()

            self.scaler.scale(l_total).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(
                list(self.student.parameters()) + list(self.feat_loss.parameters()),
                max_norm=5.0,
            )
            self.scaler.step(self.opt)
            self.scaler.update()

            losses_acc["task"].append(l_task.item())
            losses_acc["kd_scaled"].append(l_kd_scaled)
            losses_acc["kd_pre_mhsa"].append(l_kd_pre_mhsa_val)
            losses_acc["feat_scaled"].append(l_feat_scaled)
            losses_acc["phys_scaled"].append(l_phys_scaled)
            losses_acc["total"].append(l_total.item())

            pbar.set_postfix({
                "task": f"{l_task.item():.4f}",
                "kd": f"{l_kd_scaled:.4f}",
                "feat": f"{l_feat_scaled:.4f}",
                "phys": f"{l_phys_scaled:.4f}",
            })

        return {k: np.mean(v) for k, v in losses_acc.items()}

    @torch.no_grad()
    def validate(self, loader, epoch):
        self.student.eval()

        all_S_prob, all_S_true = [], []
        all_P_prob, all_P_true = [], []
        ics_P_prob, ics_P_true = [], []       # primary metrics on ICS events only
        ics_S_prob, ics_S_true = [], []       # scatter metrics on ICS events only
        all_ics_prob, all_ics_true = [], []
        top1_correct, top1_total = 0, 0

        for x, y in loader:
            x = x.to(self.device)
            y = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in y.items()}

            out = self.student(x)
            S_prob = torch.sigmoid(out["S_logits"]).squeeze(1).cpu()
            P_prob = torch.sigmoid(out["P_logits"]).squeeze(1).cpu()
            ics_prob = torch.sigmoid(out["ics_logit"]).cpu()
            is_ics = y["is_ics"].cpu()

            # AccTop1: is the highest-scoring pixel a true primary?
            P_flat = P_prob.view(P_prob.shape[0], -1)
            P_true_flat = y["P_star"].cpu().view(P_prob.shape[0], -1)
            pred_idx = P_flat.argmax(dim=1)
            has_primary = P_true_flat.sum(dim=1) > 0
            gt_at_pred = P_true_flat[torch.arange(len(pred_idx)), pred_idx]
            top1_correct += int((gt_at_pred[has_primary] > 0).sum())
            top1_total += int(has_primary.sum())

            all_S_prob.append(S_prob.reshape(-1))
            all_S_true.append(y["S_star"].cpu().reshape(-1))
            all_P_prob.append(P_prob.reshape(-1))
            all_P_true.append(y["P_star"].cpu().reshape(-1))
            all_ics_prob.append(ics_prob)
            all_ics_true.append(is_ics.float())

            # Collect primary / scatter predictions only for ICS events
            for b in range(x.shape[0]):
                if is_ics[b]:
                    ics_P_prob.append(P_prob[b].flatten())
                    ics_P_true.append(y["P_star"][b].cpu().flatten())
                    ics_S_prob.append(S_prob[b].flatten())
                    ics_S_true.append(y["S_star"][b].cpu().flatten())

        S_prob = torch.cat(all_S_prob).numpy()
        S_true = torch.cat(all_S_true).numpy()
        P_prob = torch.cat(all_P_prob).numpy()
        P_true = torch.cat(all_P_true).numpy()
        ics_prob = torch.cat(all_ics_prob).numpy()
        ics_true = torch.cat(all_ics_true).numpy()
        ics_P_prob_np = torch.cat(ics_P_prob).numpy() if ics_P_prob else np.array([])
        ics_P_true_np = torch.cat(ics_P_true).numpy() if ics_P_true else np.array([])
        ics_S_prob_np = torch.cat(ics_S_prob).numpy() if ics_S_prob else np.array([])
        ics_S_true_np = torch.cat(ics_S_true).numpy() if ics_S_true else np.array([])

        def best_threshold_f1(probs, labels):
            """Find threshold maximizing F1 via precision-recall curve."""
            precision, recall, thresholds = precision_recall_curve(labels, probs)
            f1s = 2 * precision * recall / np.maximum(precision + recall, 1e-8)
            best_idx = np.argmax(f1s)
            best_thr = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
            return float(f1s[best_idx]), float(best_thr)

        metrics = {}

        # Scatter metrics
        if S_true.sum() > 0:
            metrics["AUPRC_Scatter"] = average_precision_score(S_true, S_prob)
            metrics["AUROC_Scatter"] = roc_auc_score(S_true, S_prob)
            best_f1, best_thr = best_threshold_f1(S_prob, S_true)
            metrics["Scatter_F1"] = best_f1
            metrics["Scatter_Thr"] = best_thr

        # Primary metrics (all events — diluted by 92% non-ICS)
        if P_true.sum() > 0:
            metrics["AUPRC_Primary"] = average_precision_score(P_true, P_prob)
            best_f1, best_thr = best_threshold_f1(P_prob, P_true)
            metrics["Primary_F1"] = best_f1
            metrics["Primary_Thr"] = best_thr
        if top1_total > 0:
            metrics["Primary_AccTop1"] = top1_correct / top1_total

        # Primary metrics on ICS events only (operationally relevant)
        if len(ics_P_true_np) > 0 and ics_P_true_np.sum() > 0:
            metrics["AUPRC_Primary_ICS"] = average_precision_score(ics_P_true_np, ics_P_prob_np)
            best_f1_ics, best_thr_ics = best_threshold_f1(ics_P_prob_np, ics_P_true_np)
            metrics["Primary_F1_ICS"] = best_f1_ics

        # Scatter metrics on ICS events only (excludes non-ICS dilution)
        if len(ics_S_true_np) > 0 and ics_S_true_np.sum() > 0:
            metrics["AUPRC_Scatter_ICS"] = average_precision_score(ics_S_true_np, ics_S_prob_np)
            best_f1_s_ics, best_thr_s_ics = best_threshold_f1(ics_S_prob_np, ics_S_true_np)
            metrics["Scatter_F1_ICS"] = best_f1_s_ics
            metrics["Scatter_Thr_ICS"] = best_thr_s_ics

        # ICS metrics
        if ics_true.sum() > 0 and (1 - ics_true).sum() > 0:
            metrics["AUPRC_ICS"] = average_precision_score(ics_true, ics_prob)
            metrics["AUROC_ICS"] = roc_auc_score(ics_true, ics_prob)
            best_f1, best_thr = best_threshold_f1(ics_prob, ics_true)
            metrics["ICS_F1"] = best_f1
            metrics["ICS_Thr"] = best_thr

        # Composite score: primary localization only (scatter/ICS not our objective)
        metrics["composite"] = metrics.get("AUPRC_Primary", 0.0)

        return metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        state = {
            "epoch": epoch,
            "student": self.student.state_dict(),
            "feat_proj": self.feat_loss.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "cfg": self.cfg,
        }
        torch.save(state, os.path.join(self.ckpt_dir, "last.pt"))
        if is_best:
            torch.save(state, os.path.join(self.ckpt_dir, "best.pt"))

    def load_checkpoint(self, ckpt_path: str) -> int:
        """
        Load student, feat_proj, optimizer, and scaler state from a checkpoint.
        Returns the saved epoch number.

        Usage: call before fit() to resume training. Pass start_epoch=epoch+1
        to fit() so the curriculum picks up from the right phase.
        """
        state = torch.load(ckpt_path, map_location=self.device)
        self.student.load_state_dict(state["student"])
        self.feat_loss.load_state_dict(state["feat_proj"])
        self.opt.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.best_score = state["metrics"].get("composite", -1.0)
        saved_epoch = state.get("epoch", 0)
        # Reset LR to base so each resumed phase starts fresh (avoids
        # inheriting a decayed LR from the previous phase's ReduceLROnPlateau)
        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr
        print(f"  Resumed from epoch {saved_epoch} "
              f"(best_score={self.best_score:.4f}, LR reset to {self.base_lr:.2e})")
        return saved_epoch

    def fit(self, dl_train, dl_val, epochs=30, early_patience=7, start_epoch=0):
        n_teachers = len(self.teachers) if self.teachers else 0
        print(f"\n{'='*70}")
        print(f"PI-KD Training: {epochs} epochs")
        print(f"  Curriculum:        {self.curriculum}")
        print(f"  Student params:    {self.student.count_params():,}")
        print(f"  Teachers:          {n_teachers} {'(ensemble)' if n_teachers > 1 else ''}")
        print(f"  Feature matching:  {'ON' if self.use_feature_matching else 'OFF'}")
        pre_w, post_w = self.pre_mhsa_weight, self.post_mhsa_weight
        print(f"  KD mix:            pre-MHSA={pre_w:.0%} / post-MHSA={post_w:.0%}")
        print(f"  Temp annealing:    {self.curriculum.T_start} -> {self.curriculum.T_end}")
        print(f"{'='*70}\n")

        patience_counter = 0
        prev_phase = None
        phase_start_epoch = start_epoch

        for epoch in range(start_epoch, epochs):
            t0 = time.time()
            lambdas = self.curriculum.get_lambdas(epoch)
            phase = self.curriculum.get_phase_name(epoch)

            if phase != prev_phase and prev_phase is not None:
                print(f"\n  [Phase transition: {prev_phase} -> {phase}] "
                      f"Resetting patience, LR, and applying warmup.")
                patience_counter = 0
                phase_start_epoch = epoch
                for pg in self.opt.param_groups:
                    pg["lr"] = self.base_lr
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.opt, mode="max", factor=0.5, patience=5,
                )
            prev_phase = phase

            self._apply_warmup_lr(epoch, phase_start_epoch)

            # Train
            train_losses = self.train_one_epoch(dl_train, epoch)

            # Validate
            val_metrics = self.validate(dl_val, epoch)
            score = val_metrics["composite"]
            if (epoch - phase_start_epoch) >= self.warmup_epochs:
                self.scheduler.step(score)

            # Checkpoint
            is_best = score > self.best_score
            if is_best:
                self.best_score = score
                patience_counter = 0
            else:
                patience_counter += 1

            self.save_checkpoint(epoch, val_metrics, is_best=is_best)

            # Log
            dt = time.time() - t0
            lr = self.opt.param_groups[0]["lr"]
            T = lambdas["temperature"]
            print(
                f"Ep {epoch:3d} [{phase:16s}] T={T:4.1f} "
                f"loss={train_losses['total']:.4f} "
                f"(task={train_losses['task']:.4f} kd={train_losses['kd_scaled']:.4f} "
                f"kd_pre={train_losses['kd_pre_mhsa']:.4f} "
                f"feat={train_losses['feat_scaled']:.4f} phys={train_losses['phys_scaled']:.4f}) | "
                f"S_F1={val_metrics.get('Scatter_F1', 0):.3f}(ap={val_metrics.get('AUPRC_Scatter', 0):.3f}) "
                f"S_ICS={val_metrics.get('Scatter_F1_ICS', 0):.3f}(ap={val_metrics.get('AUPRC_Scatter_ICS', 0):.3f}) "
                f"P_F1={val_metrics.get('Primary_F1', 0):.3f}(ap={val_metrics.get('AUPRC_Primary', 0):.3f}) "
                f"P_ICS={val_metrics.get('Primary_F1_ICS', 0):.3f}(ap={val_metrics.get('AUPRC_Primary_ICS', 0):.3f}) "
                f"AccTop1={val_metrics.get('Primary_AccTop1', 0):.4f} "
                f"ICS_F1={val_metrics.get('ICS_F1', 0):.3f}(ap={val_metrics.get('AUPRC_ICS', 0):.3f}) "
                f"score={score:.4f} {'*' if is_best else ''} "
                f"lr={lr:.2e} {dt:.0f}s"
            )

            if patience_counter >= early_patience:
                print(f"\nEarly stopping at epoch {epoch} (patience={early_patience})")
                break

        print(f"\nTraining complete. Best composite score: {self.best_score:.4f}")
        print(f"Checkpoints saved to {self.ckpt_dir}/")
