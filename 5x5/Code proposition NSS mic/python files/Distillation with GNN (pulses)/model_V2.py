
import os, math, random, json
from pathlib import Path
import numpy as np
import pandas as pd
from torch.amp import GradScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
#from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, precision_recall_curve

import matplotlib.pyplot as plt

class PhysicsBiasedMHSA(nn.Module):
    def __init__(self, d=96, heads=4, c_mm_per_ns=299.792458, sigma_t_ns=0.2,
                 alpha_tof=2.0, alpha_spatial=0.1, confinement_radius_px=2.0,
                 pitch_xy_mm=(2.0,2.0)):
        super().__init__()
        assert d % heads == 0
        self.d = d; self.h = heads; self.dk = d // heads
        self.Wq = nn.Linear(d, d); self.Wk = nn.Linear(d, d); self.Wv = nn.Linear(d, d)
        self.proj = nn.Linear(d, d)
        self.ffn  = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4*d), nn.ReLU(), nn.Linear(4*d, d))
        self.ln   = nn.LayerNorm(d)
        self.c = c_mm_per_ns; self.sigma_t = sigma_t_ns
        self.alpha_tof = alpha_tof; self.alpha_sp = alpha_spatial
        self.R = confinement_radius_px
        self.pitch_x, self.pitch_y = pitch_xy_mm
        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing='ij')
        idx = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
        self.register_buffer("idx_px", idx)
        dxy_px = idx.unsqueeze(1) - idx.unsqueeze(0)
        dist_px = torch.sqrt((dxy_px[...,0]**2 + dxy_px[...,1]**2))
        self.register_buffer("dist_px", dist_px)
        dx_mm = dxy_px[...,1]*self.pitch_x; dy_mm = dxy_px[...,0]*self.pitch_y
        dist_mm = torch.sqrt(dx_mm**2 + dy_mm**2)
        self.register_buffer("dist_mm", dist_mm)
        self.register_buffer("hard_mask", (dist_px > self.R))
    def build_bias(self, T_hat):
        B, N, _ = T_hat.shape
        dt = T_hat - T_hat.transpose(1,2)
        dt = torch.clamp(dt, -10*self.sigma_t, 10*self.sigma_t)
        tof = -self.alpha_tof * ((dt - self.dist_mm/self.c) / (self.sigma_t + 1e-9))**2
        spa = -self.alpha_sp * (self.dist_px**2).unsqueeze(0).expand(B,-1,-1)
        bias = torch.clamp(tof + spa, -50.0, 50.0)
        bias = bias.masked_fill(self.hard_mask.unsqueeze(0), float('-inf'))
        return bias.unsqueeze(1).to(T_hat.dtype)
    def forward(self, tok, T_hat):
        B, N, D = tok.shape
        q = self.Wq(tok).view(B, N, self.h, self.dk).transpose(1,2)
        k = self.Wk(tok).view(B, N, self.h, self.dk).transpose(1,2)
        v = self.Wv(tok).view(B, N, self.h, self.dk).transpose(1,2)
        bias = self.build_bias(T_hat)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        attn = attn.transpose(1,2).contiguous().view(B, N, D)
        out  = self.proj(attn)
        out  = self.ln(out + tok)
        out  = out + self.ffn(out)
        return out

class DenseHeads(nn.Module):
    def __init__(self, d=96):
        super().__init__()
        self.convS = nn.Conv2d(d, 1, 1)  # scatter
        self.convP = nn.Conv2d(d, 1, 1)  # primary

    def forward(self, feat):
        S_logits = self.convS(feat)
        P_logits = self.convP(feat)
        return S_logits, P_logits


class ICSHead(nn.Module):
    """
    Pulse-level binary classifier: predicts P(ICS).
    Enriched with 6 explicit physics features:
      - s_max      : max scatter probability across pixels  (ICS → high)
      - s_mean     : mean scatter probability across pixels (ICS → higher)
      - e_var      : variance of energy map                 (ICS → higher)
      - n_active   : fraction of active pixels              (ICS → more active)
      - e_max_frac : max_energy / total_energy              (ICS → lower, energy split)
      - t_spread   : time range among active pixels         (ICS → higher, scatter delayed)
    """
    def __init__(self, d=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 6, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1)
        )

    def forward(self, feat2, S_logits, E_input):
        # feat2:   (B, d, 5, 5)
        # S_logits:(B, 1, 5, 5)
        # E_input: (B, C, 5, 5) — ch0=energy, ch1=time, ch2=active_mask, ch3+=extra
        pooled = feat2.mean(dim=[2, 3])                               # (B, d)

        s_prob = torch.sigmoid(S_logits).flatten(1)                   # (B, 25)
        s_max  = s_prob.max(dim=1, keepdim=True).values               # (B, 1)
        s_mean = s_prob.mean(dim=1, keepdim=True)                     # (B, 1)

        e_flat = E_input[:, 0].flatten(1)                             # (B, 25) energy
        e_var  = e_flat.var(dim=1, keepdim=True)                      # (B, 1)

        # Physics features
        active = E_input[:, 2].flatten(1)                             # (B, 25) binary mask
        n_active = active.sum(dim=1, keepdim=True) / 25.0            # (B, 1) normalised

        e_sum = e_flat.sum(dim=1, keepdim=True).clamp(min=1e-6)
        e_max_frac = e_flat.max(dim=1, keepdim=True).values / e_sum  # (B, 1)

        t_flat = E_input[:, 1].flatten(1)                             # (B, 25) time (shifted)
        t_spread = (t_flat * active).max(dim=1, keepdim=True).values  # (B, 1) since min=0

        x = torch.cat([pooled, s_max, s_mean, e_var,
                        n_active, e_max_frac, t_spread], dim=1)       # (B, d+6)
        return self.net(x).squeeze(-1)                                # (B,) logits


class GraphPrimaryHead(nn.Module):
    def __init__(self, d=96, heads=4, layers=1, topk=10, dropout=0.2, g_dim=None):
        super().__init__()
        self.topk = topk

        in_dim = d + 1 + 2   # feat(d) + cand_score(1) + xy(2)
        self.g_dim = g_dim if g_dim is not None else d  # ex: 96
        assert self.g_dim % heads == 0, "g_dim must be divisible by heads"

        self.in_proj = nn.Linear(in_dim, self.g_dim)

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.g_dim, nhead=heads, batch_first=True,
                dropout=dropout, dim_feedforward=4*self.g_dim
            )
            for _ in range(layers)
        ])
        self.cls = nn.Sequential(nn.LayerNorm(self.g_dim), nn.Linear(self.g_dim, 1))

        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing='ij')
        self.register_buffer("xy_idx", torch.stack([xx, yy], dim=-1).view(-1, 2).float())

    def forward(self, feat_5x5, cand_score):
        B, D, H, W = feat_5x5.shape; N = H * W
        feat = feat_5x5.flatten(2).transpose(1, 2)          # (B,N,d)
        s    = cand_score.flatten(2).transpose(1, 2)        # (B,N,1)
        xy   = self.xy_idx.unsqueeze(0).expand(B, -1, -1)   # (B,N,2)

        K = min(self.topk, N)
        k_idx = torch.topk(s.squeeze(-1), k=K, dim=1).indices

        def gather(x):  # (B,N,F) -> (B,K,F)
            return x.gather(1, k_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1]))

        nodes = torch.cat([gather(feat), gather(s), gather(xy)], dim=-1)  # (B,K,in_dim)
        z = self.in_proj(nodes)                                           # (B,K,g_dim)
        for l in self.layers:
            z = l(z)
        logitsK = self.cls(z).squeeze(-1)                                 # (B,K)
        return logitsK, k_idx



class AssignmentHead(nn.Module):
    """
    For each scatter pixel, predict which primary it belongs to.
    Output: (B, K_max_primaries, 5, 5) — soft assignment probabilities.
    For simplicity, we predict a (B, 1, 5, 5) offset map pointing toward the
    assigned primary, then match via nearest-primary logic at eval time.

    Actually simpler: predict (B, 2, 5, 5) — (dy, dx) offset from each scatter
    pixel to its assigned primary. The loss penalises distance to GT primary.
    """
    def __init__(self, d=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(d, d // 2, 1), nn.ReLU(),
            nn.Conv2d(d // 2, 2, 1),  # predict (dy, dx) offsets
        )
        yy, xx = torch.meshgrid(torch.arange(5), torch.arange(5), indexing='ij')
        self.register_buffer("grid_coords", torch.stack([yy, xx], dim=0).float())  # (2, 5, 5)

    def forward(self, feat, S_prob):
        """
        Args:
            feat: (B, d, 5, 5) features from MHSA
            S_prob: (B, 1, 5, 5) scatter probability (detached)
        Returns:
            offsets: (B, 2, 5, 5) predicted (dy, dx) offsets
            pred_targets: (B, 2, 5, 5) predicted primary coordinates (pixel_y, pixel_x)
        """
        offsets = self.net(feat)  # (B, 2, 5, 5)
        # Predicted primary location = current pixel + offset
        pred_targets = self.grid_coords.unsqueeze(0) + offsets  # (B, 2, 5, 5)
        return offsets, pred_targets


class CorrectionHead(nn.Module):
    """
    Predicts the scatter-corrected energy map.
    Input: features + original energy channel + scatter probability.
    Output: (B, 1, 5, 5) corrected energy.
    """
    def __init__(self, d=96):
        super().__init__()
        # d features + 1 energy + 1 scatter_prob = d+2
        self.net = nn.Sequential(
            nn.Conv2d(d + 2, d // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(d // 2, 1, 1),
            nn.ReLU(),  # energy is non-negative
        )

    def forward(self, feat, E_input, S_prob):
        """
        Args:
            feat: (B, d, 5, 5)
            E_input: (B, 1, 5, 5) original energy channel
            S_prob: (B, 1, 5, 5) scatter probability
        Returns:
            E_corrected: (B, 1, 5, 5)
        """
        x = torch.cat([feat, E_input, S_prob.detach()], dim=1)
        return self.net(x)


class ExampleTrunk(nn.Module):
    def __init__(self, in_ch=6, out_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, out_ch, 3, padding=1),nn.BatchNorm2d(out_ch), nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class PhysFormerWrapper(nn.Module):
    def __init__(self, trunk: nn.Module, d=96, heads=4, mhsa_layers=2,
                 topk=10, primary_cap=0.38, primary_cap_tol=0.02,
                 c_mm_per_ns=299.792458, sigma_t_ns=0.2,
                 alpha_tof=2.0, alpha_spatial=0.1,
                 confinement_radius_px=2.0, pitch_xy_mm=(2.0,2.0),
                 in_ch=6):
        super().__init__()
        self.trunk = trunk
        self.d = d
        self.in_ch = in_ch

        was_training = self.trunk.training
        self.trunk.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, 5, 5)
            c_out = self.trunk(dummy).shape[1]
        if was_training:
            self.trunk.train()

        self.proj = nn.Identity() if c_out == d else nn.Conv2d(c_out, d, 1)

        self.mhsas = nn.ModuleList([
            PhysicsBiasedMHSA(d, heads, c_mm_per_ns, sigma_t_ns,
                              alpha_tof, alpha_spatial,
                              confinement_radius_px, pitch_xy_mm)
            for _ in range(mhsa_layers)
        ])
        self.heads      = DenseHeads(d)
        self.graph      = GraphPrimaryHead(d, heads, layers=1, topk=topk)
        self.ics_head   = ICSHead(d)
        self.assign_head = AssignmentHead(d)
        self.correction_head = CorrectionHead(d)

        self.cap = primary_cap
        self.cap_tol = primary_cap_tol

    def forward(self, x, use_T_bias=None, learn_bias: bool = False):
        feat0 = self.trunk(x)
        feat  = self.proj(feat0)

        # pré-têtes (sans E/T)
        S1_logits, P1_logits = self.heads(feat)
        # pour le biais temporel des MHSA, on n’a plus de T -> on met zéro
        T_bias = torch.zeros_like(S1_logits) if use_T_bias is None else use_T_bias

        tok   = feat.flatten(2).transpose(1,2)
        T_hat = T_bias.flatten(2).transpose(1,2)

        for blk in self.mhsas:
            tok = blk(tok, T_hat if learn_bias else T_hat.detach())

        feat2 = tok.transpose(1,2).view_as(feat)

        S_logits, P_logits = self.heads(feat2)
        S = torch.sigmoid(S_logits)
        P = torch.sigmoid(P_logits)

        # candidats = "primaire et non-scatter"
        cand_score = torch.clamp(P * (1 - torch.sigmoid(S_logits)), 0, 1)

        logitsK, idxK = self.graph(feat2, cand_score)

        # ICS gate: pulse-level P(ICS) — enriched with scatter & energy features
        ics_logit = self.ics_head(feat2, S_logits.detach(), x)        # (B,)
        ics_log_prob = F.logsigmoid(ics_logit)                       # log P(ICS), (B,)
        logitsK_gated = logitsK + ics_log_prob.unsqueeze(1)          # (B,K)

        # Assignment: scatter pixel → primary offset
        assign_offsets, assign_targets = self.assign_head(feat2, S.detach())

        # Correction: predict scatter-free energy map
        E_input = x[:, 0:1, :, :]  # first channel is energy
        E_corrected = self.correction_head(feat2, E_input, S.detach())

        return {
            "S_logits": S_logits,
            "S": S,
            "P_logits": P_logits,
            "P": P,
            "primary_logits": logitsK_gated,      # gated — for evaluation
            "primary_logits_raw": logitsK,         # raw — for ranking loss training
            "primary_idx": idxK,
            "ics_logit": ics_logit,
            "assign_offsets": assign_offsets,       # (B, 2, 5, 5)
            "assign_targets": assign_targets,       # (B, 2, 5, 5)
            "E_corrected": E_corrected,             # (B, 1, 5, 5)
        }
