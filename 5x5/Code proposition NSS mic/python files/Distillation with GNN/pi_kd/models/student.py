"""
StudentNet: FPGA-deployable architecture for primary pixel localization.

Mono-head version (2026-05-05): only primary_head is trained and exported.
  - scatter_head removed (was loss_weight=0.15 regularizer, not exported)
  - ics_head removed (was loss_weight=0.0, completely unused)

Constraints:
  - No attention / softmax (too costly in fixed-point)
  - Depthwise separable convolutions only
  - ReLU6 activations only
  - Fixed input: (B, 3, 5, 5) = [energy, time, active_mask]

Output:
  - P_logits: (B, 1, 5, 5) primary pixel probability map
"""

import torch
import torch.nn as nn

from .layers import DepthwiseSepBlock


class StudentNet(nn.Module):

    def __init__(self, in_ch: int = 3, widths: tuple = (32, 64, 64)):
        super().__init__()
        self.in_ch = in_ch

        layers = []
        c_in = in_ch
        for c_out in widths:
            layers.append(DepthwiseSepBlock(c_in, c_out))
            c_in = c_out
        self.backbone = nn.Sequential(*layers)
        self.feat_dim = widths[-1]

        self.primary_head = nn.Conv2d(self.feat_dim, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (B, 3, 5, 5)
        Returns:
            dict with P_logits, feat, feat_inter
        """
        feat_inter = self.backbone[0](x)
        feat_inter = self.backbone[1](feat_inter)
        feat = self.backbone[2](feat_inter)

        P_logits = self.primary_head(feat)

        return {
            "P_logits": P_logits,
            "feat": feat,
            "feat_inter": feat_inter,
        }

    def fuse_bn(self):
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
        total = 0
        B, C, H, W = input_shape
        c_in = C
        for block in self.backbone:
            total += c_in * 9 * H * W
            c_out = block.pw_conv.out_channels
            total += c_in * c_out * H * W
            c_in = c_out
        total += c_in * 1 * H * W  # primary head
        return total
