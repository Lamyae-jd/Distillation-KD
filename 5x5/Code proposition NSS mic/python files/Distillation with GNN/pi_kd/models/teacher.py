"""
Teacher wrapper: loads the frozen PhysFormerWrapper and exposes
a clean inference interface for distillation.

Supports:
  - Single teacher loading
  - Multi-teacher ensemble (average logits from multiple checkpoints)
  - Intermediate feature extraction for layer-wise distillation
"""

import sys
import os
from typing import List, Optional
import torch
import torch.nn as nn

# Add parent directory to path to import the teacher model
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)


def _build_teacher_from_ckpt(ckpt: dict, device: str = "cpu") -> nn.Module:
    """Build a PhysFormerWrapper from a checkpoint dict."""
    from model_V2 import ExampleTrunk, PhysFormerWrapper

    cfg = ckpt["cfg"]
    trunk = ExampleTrunk(in_ch=3, out_ch=64)
    model = PhysFormerWrapper(
        trunk,
        d=cfg["model"]["embed_dim"],
        heads=cfg["model"]["heads"],
        mhsa_layers=cfg["model"]["mhsa_layers"],
        topk=cfg["model"]["topk"],
        primary_cap=cfg["model"]["primary_cap"],
        primary_cap_tol=cfg["model"]["primary_cap_tol"],
        c_mm_per_ns=cfg["model"]["c_mm_per_ns"],
        sigma_t_ns=cfg["model"]["sigma_t_ns"],
        alpha_tof=cfg["model"]["alpha_tof"],
        alpha_spatial=cfg["model"]["alpha_spatial"],
        confinement_radius_px=cfg["model"]["confinement_radius_px"],
        in_ch=3,
    )
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_teacher(checkpoint_path: str, device: str = "cpu") -> nn.Module:
    """
    Load a single trained PhysFormerWrapper from a checkpoint.

    Args:
        checkpoint_path: path to best.pt or last.pt
        device: target device

    Returns:
        Frozen PhysFormerWrapper in eval mode.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return _build_teacher_from_ckpt(ckpt, device)


def load_multi_teacher(checkpoint_paths: List[str], device: str = "cpu") -> List[nn.Module]:
    """
    Load multiple teacher checkpoints for ensemble distillation.

    Uses different checkpoints (different epochs, seeds, or runs) to produce
    a smoother, more robust supervision signal by averaging their logits.

    Args:
        checkpoint_paths: list of paths to checkpoint files
        device: target device

    Returns:
        List of frozen PhysFormerWrapper models.
    """
    teachers = []
    for path in checkpoint_paths:
        if not os.path.exists(path):
            print(f"  WARNING: teacher checkpoint not found: {path}, skipping")
            continue
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = _build_teacher_from_ckpt(ckpt, device)
        teachers.append(model)
        print(f"  Loaded teacher from {path}")
    if not teachers:
        raise FileNotFoundError(f"No valid teacher checkpoints found in {checkpoint_paths}")
    print(f"  Multi-teacher ensemble: {len(teachers)} models")
    return teachers


def _extract_trunk_features(teacher: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract intermediate features from the teacher's CNN trunk (after layer 2).

    ExampleTrunk.net = Sequential(
        Conv2d(3,64) -> BN -> ReLU,   # layers 0,1,2
        Conv2d(64,64) -> BN -> ReLU,  # layers 3,4,5  <-- extract after this
        Conv2d(64,64) -> BN -> ReLU,  # layers 6,7,8
    )

    Returns:
        Feature map (B, 64, 5, 5) after the second conv block.
    """
    trunk_net = teacher.trunk.net
    # Run through first 6 layers (2 complete conv+bn+relu blocks)
    feat = x
    for i in range(6):
        feat = trunk_net[i](feat)
    return feat


@torch.no_grad()
def teacher_inference(teacher: nn.Module, x: torch.Tensor,
                      T_star: torch.Tensor = None,
                      extract_features: bool = False) -> dict:
    """
    Run teacher forward pass and return distillation-relevant outputs.

    Args:
        x: (B, 3, 5, 5) input batch
        T_star: (B, 1, 5, 5) timing map for MHSA physics bias.
                CRITICAL: without this, the teacher's attention has no
                temporal information and produces degraded predictions.
        extract_features: if True, also return intermediate trunk features

    Returns:
        dict with S_logits, P_logits, P1_logits, ics_logit (all detached),
        and optionally 'feat_inter' (B, 64, 5, 5)
    """
    out = teacher(x, use_T_bias=T_star)
    result = {
        "S_logits": out["S_logits"].detach(),
        "P_logits": out["P_logits"].detach(),
        "ics_logit": out["ics_logit"].detach(),
    }
    if "P1_logits" in out:
        result["P1_logits"] = out["P1_logits"].detach()
    if extract_features:
        result["feat_inter"] = _extract_trunk_features(teacher, x).detach()
    return result


@torch.no_grad()
def multi_teacher_inference(teachers: List[nn.Module], x: torch.Tensor,
                            T_star: torch.Tensor = None,
                            extract_features: bool = False) -> dict:
    """
    Run multiple teachers and average their logits.

    Produces a smoother supervision signal than any single teacher.
    Each teacher's logits are averaged element-wise before returning.

    Args:
        teachers: list of frozen PhysFormerWrapper models
        x: (B, 3, 5, 5) input batch
        T_star: (B, 1, 5, 5) timing map for MHSA physics bias
        extract_features: if True, also average intermediate features

    Returns:
        dict with averaged S_logits, P_logits, ics_logit,
        and optionally averaged 'feat_inter'
    """
    n = len(teachers)
    S_acc, P_acc, P1_acc, ics_acc = None, None, None, None
    feat_acc = None

    has_p1 = True  # confirmed at first teacher; left as flag for clarity
    for i, t in enumerate(teachers):
        out = t(x, use_T_bias=T_star)
        s = out["S_logits"].detach()
        p = out["P_logits"].detach()
        ic = out["ics_logit"].detach()
        if "P1_logits" in out:
            p1 = out["P1_logits"].detach()
            P1_acc = p1 if P1_acc is None else P1_acc + p1
        else:
            has_p1 = False

        S_acc = s if S_acc is None else S_acc + s
        P_acc = p if P_acc is None else P_acc + p
        ics_acc = ic if ics_acc is None else ics_acc + ic

        if extract_features:
            f = _extract_trunk_features(t, x).detach()
            feat_acc = f if feat_acc is None else feat_acc + f

    result = {
        "S_logits": S_acc / n,
        "P_logits": P_acc / n,
        "ics_logit": ics_acc / n,
    }
    if has_p1 and P1_acc is not None:
        result["P1_logits"] = P1_acc / n
    if extract_features and feat_acc is not None:
        result["feat_inter"] = feat_acc / n

    return result
