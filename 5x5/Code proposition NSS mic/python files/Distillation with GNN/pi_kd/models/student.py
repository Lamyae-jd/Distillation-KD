"""
StudentNet: FPGA-deployable architecture for ICS detection.

Constraints:
  - No attention / softmax (too costly in fixed-point)
  - Depthwise separable convolutions only
  - ReLU6 activations only
  - Fixed input: (B, 3, 5, 5) = [energy, time, active_mask]

Outputs:
  - S_logits: (B, 1, 5, 5) scatter probability map
  - P_logits: (B, 1, 5, 5) primary pixel probability map
  - ics_logit: (B,) pulse-level ICS classification
"""

import torch
import torch.nn as nn

from .layers import DepthwiseSepBlock


class StudentNet(nn.Module):

    def __init__(self, in_ch: int = 3, widths: tuple = (32, 64, 64)):
        super().__init__()
        self.in_ch = in_ch

        # Backbone: 3 depthwise separable blocks
        layers = []
        c_in = in_ch
        for c_out in widths:
            layers.append(DepthwiseSepBlock(c_in, c_out))
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.feat_dim = widths[-1]

        # --- Dense heads (pixel-level) ---
        self.scatter_head = nn.Conv2d(self.feat_dim, 1, kernel_size=1, bias=True)
        self.primary_head = nn.Conv2d(self.feat_dim, 1, kernel_size=1, bias=True)

        # --- ICS head (pulse-level) ---
        # Global average pooling + small MLP with ReLU6
        self.ics_head = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim // 2),
            nn.ReLU6(inplace=True),
            nn.Linear(self.feat_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 5, 5)
        Returns:
            dict with S_logits, P_logits, ics_logit, feat, feat_inter
        """
        # Extract intermediate features after block 1 (for layer-wise distillation)
        feat_inter = self.backbone[0](x)     # (B, 32, 5, 5) after block 0
        feat_inter = self.backbone[1](feat_inter)  # (B, 64, 5, 5) after block 1

        # Final features
        feat = self.backbone[2](feat_inter)  # (B, 64, 5, 5) after block 2

        S_logits = self.scatter_head(feat)   # (B, 1, 5, 5)
        P_logits = self.primary_head(feat)   # (B, 1, 5, 5)

        # Global average pooling -> ICS classification
        pooled = feat.mean(dim=[2, 3])       # (B, 64)
        ics_logit = self.ics_head(pooled).squeeze(-1)  # (B,)

        return {
            "S_logits": S_logits,
            "P_logits": P_logits,
            "ics_logit": ics_logit,
            "feat": feat,        # final backbone features
            "feat_inter": feat_inter,  # intermediate features for layer-wise distillation
        }

    def fuse_bn(self):
        """Fuse BatchNorm into Conv layers for deployment."""
        torch.quantization.fuse_modules(
            self.backbone[0], [["dw_conv", "dw_bn"], ["pw_conv", "pw_bn"]], inplace=True
        )
        torch.quantization.fuse_modules(
            self.backbone[1], [["dw_conv", "dw_bn"], ["pw_conv", "pw_bn"]], inplace=True
        )
        torch.quantization.fuse_modules(
            self.backbone[2], [["dw_conv", "dw_bn"], ["pw_conv", "pw_bn"]], inplace=True
        )

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def count_flops(self, input_shape=(1, 3, 5, 5)) -> int:
        """Approximate FLOPs for a single forward pass."""
        total = 0
        B, C, H, W = input_shape
        c_in = C
        for block in self.backbone:
            # Depthwise: c_in * 3 * 3 * H * W
            total += c_in * 9 * H * W
            c_out = block.pw_conv.out_channels
            # Pointwise: c_in * c_out * H * W
            total += c_in * c_out * H * W
            c_in = c_out
        # Heads: 1x1 convs
        total += c_in * 1 * H * W  # scatter
        total += c_in * 1 * H * W  # primary
        # ICS MLP
        total += c_in * (c_in // 2) + (c_in // 2) * 1
        return total
