"""
Knowledge Distillation losses:

1. L_KD: KL divergence between teacher and student soft distributions
   at dynamic temperature T (annealed from T_start=10 to T_end=2).
   L_KD = BCE(sigmoid(z_s / T), sigmoid(z_t / T)) * T^2

2. L_feat: Feature matching loss between intermediate representations.
   Uses a learned 1x1 projection to align student features to teacher space.
   L_feat = MSE(proj(student_feat_inter), teacher_feat_inter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    KL divergence on softened logits with dynamic temperature.

    Temperature is passed per-call (from CurriculumScheduler) to support
    temperature annealing: T starts at 2.0 and decays to 1.2.
    For binary sigmoid KD, T must stay low to preserve soft-label contrast.
    """

    def __init__(self, temperature: float = 1.5):
        super().__init__()
        self.T_default = temperature

    def binary_kl(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  T: float = None) -> torch.Tensor:
        """
        Binary KL divergence with temperature scaling.

        Args:
            student_logits: raw logits from student
            teacher_logits: raw logits from teacher
            T: temperature (if None, uses self.T_default)
        """
        T = T if T is not None else self.T_default
        s_logits = student_logits / T
        t_logits = teacher_logits / T

        p_t = torch.sigmoid(t_logits)
        loss = F.binary_cross_entropy_with_logits(s_logits, p_t, reduction="mean")
        # Note: T² compensation is omitted for binary BCE.
        # The Hinton T² trick applies to multi-class KL/softmax where gradients
        # shrink by 1/T². For binary sigmoid, T² scaling causes ~100x loss
        # inflation at T=10 which overwhelms task learning entirely.
        return loss

    def forward(self, student_out: dict, teacher_out: dict,
                temperature: float = None,
                pre_mhsa_weight: float = 0.0,
                post_mhsa_weight: float = 1.0) -> dict:
        """
        Compute distillation loss on primary head only.

        When pre_mhsa_weight > 0 and teacher provides P1_logits (pre-MHSA),
        the KD loss is a weighted mix:
            kd_total = post_mhsa_weight * KD(P_logits) + pre_mhsa_weight * KD(P1_logits)
        This gives the student a CNN-compatible target alongside the full teacher target.

        Args:
            student_out: dict with P_logits
            teacher_out: dict with P_logits, P1_logits (optional)
            temperature: current temperature from curriculum scheduler
            pre_mhsa_weight: weight for pre-MHSA KD term (default 0.0 = disabled)
            post_mhsa_weight: weight for post-MHSA KD term (default 1.0)

        Returns:
            dict with kd_primary, kd_pre_mhsa, kd_total
        """
        T = temperature if temperature is not None else self.T_default

        kd_primary = self.binary_kl(student_out["P_logits"], teacher_out["P_logits"], T)

        kd_pre_mhsa = torch.zeros(1, device=student_out["P_logits"].device)
        if pre_mhsa_weight > 0.0 and "P1_logits" in teacher_out:
            kd_pre_mhsa = self.binary_kl(
                student_out["P_logits"], teacher_out["P1_logits"], T
            )

        kd_total = post_mhsa_weight * kd_primary + pre_mhsa_weight * kd_pre_mhsa

        return {
            "kd_primary": kd_primary,
            "kd_pre_mhsa": kd_pre_mhsa,
            "kd_total": kd_total,
        }


class FeatureMatchingLoss(nn.Module):
    """
    Layer-wise feature matching between teacher and student intermediate features.

    Uses a learned 1x1 convolution to project student features to the teacher's
    feature space, then minimizes MSE between them.

    This helps the student learn better spatial representations, not just
    mimic final output logits.

    Args:
        student_ch: number of channels in student intermediate features (default 64)
        teacher_ch: number of channels in teacher intermediate features (default 64)
    """

    def __init__(self, student_ch: int = 64, teacher_ch: int = 64):
        super().__init__()
        # Learned projection: student feature space -> teacher feature space
        # Even when dimensions match, a learned projection gives the student
        # flexibility to align its representation
        self.proj = nn.Conv2d(student_ch, teacher_ch, kernel_size=1, bias=False)

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            student_feat: (B, student_ch, 5, 5) from student backbone[1] output
            teacher_feat: (B, teacher_ch, 5, 5) from teacher trunk layer 2 output

        Returns:
            Scalar MSE loss between projected student features and teacher features
        """
        projected = self.proj(student_feat)  # (B, teacher_ch, 5, 5)
        return F.mse_loss(projected, teacher_feat)
