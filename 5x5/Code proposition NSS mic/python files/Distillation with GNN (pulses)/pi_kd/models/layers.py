"""
Depthwise Separable Conv blocks for FPGA-friendly StudentNet.
All activations are ReLU6 (bounded [0,6] for fixed-point quantization).
BatchNorm is fused into conv before export via torch.quantization.fuse_modules.
"""

import torch
import torch.nn as nn


class DepthwiseSepBlock(nn.Module):
    """
    Depthwise Separable Convolution block:
      1. Depthwise: Conv2d(groups=C_in, kernel=3x3) -> BN -> ReLU6
      2. Pointwise: Conv2d(1x1, C_in -> C_out) -> BN -> ReLU6

    ~8-9x fewer FLOPs than standard 3x3 conv for the same channel dimensions.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        # Depthwise conv: each input channel convolved independently
        self.dw_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride,
            padding=1, groups=in_ch, bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_ch)
        self.dw_relu = nn.ReLU6(inplace=True)

        # Pointwise conv: 1x1 to mix channels
        self.pw_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_ch)
        self.pw_relu = nn.ReLU6(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw_relu(self.dw_bn(self.dw_conv(x)))
        x = self.pw_relu(self.pw_bn(self.pw_conv(x)))
        return x
