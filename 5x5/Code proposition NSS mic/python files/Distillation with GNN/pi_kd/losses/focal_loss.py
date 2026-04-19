"""
Focal Loss for handling the ~1:24 class imbalance (1 primary pixel among 25).
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Binary Focal Loss with per-batch dynamic alpha.

    Args:
        gamma: focusing parameter. Higher = more focus on hard examples.
        alpha: class balance weight. If None, computed dynamically per batch.
        reduction: 'mean', 'sum', or 'none'
    """

    def __init__(self, gamma: float = 2.0, alpha: float = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, 5, 5) or (B, 25) raw logits
            targets: same shape as logits, binary {0, 1}
        """
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()

        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Dynamic alpha: proportion of negatives in this batch
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            n_pos = targets.sum().clamp(min=1)
            n_neg = (1 - targets).sum().clamp(min=1)
            alpha_pos = n_neg / (n_pos + n_neg)
            alpha_t = alpha_pos * targets + (1 - alpha_pos) * (1 - targets)

        loss = alpha_t * focal_weight * ce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
