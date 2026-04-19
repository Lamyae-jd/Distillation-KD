"""
Physics-informed loss: Klein-Nishina kinematics with incoherent scattering
function correction S(q, Zeff) for LYSO (Zeff ~ 65).

For each event classified as scatter, the energy deposition and spatial
position must satisfy Compton kinematics:

    E' = E0 / (1 + (E0 / m_e_c2) * (1 - cos(theta)))

The true differential cross-section is:
    dsigma/dOmega = dsigma_KN/dOmega * S(q, Zeff)

where S(q, Z) is the incoherent scattering function (Hubbell tables, NIST).
S(q, Z) suppresses forward scattering for high-Z materials.

q = sin(theta/2) / lambda,  lambda = 12.398 / E_keV  (in Angstroms)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.interpolate import interp1d


# --- Constants ---
M_E_C2_KEV = 510.999  # electron rest mass energy in keV
M_E_C2_MEV = 0.510999  # in MeV (matching the simulation edep units)

# Pixel pitch in mm (5x5 LYSO array)
PITCH_MM = 2.0

# Crystal thickness assumption for angle estimation (mm)
CRYSTAL_DEPTH_MM = 20.0

# --- LGSO material constants (from GateMaterials.db) ---
# LGSO: d=7.3 g/cm3, formula Lu0.62Gd1.38SiO5
# Mass fractions: Lu=0.2502, Gd=0.5005, Si=0.0648, O=0.1845
# Converted to atomic fractions via n_i = (f_i/A_i) / sum(f_j/A_j):
#   A: Lu=174.97, Gd=157.25, Si=28.09, O=16.00
LGSO_Z_LIST = [71, 64, 14, 8]           # Lu, Gd, Si, O
LGSO_FRACS  = [0.0775, 0.1725, 0.1250, 0.6250]  # atomic fractions


# --- Incoherent Scattering Function S(q, Z=65) for LYSO ---
# Approximate tabulated values from Hubbell (NIST XCOM)
# q in inverse Angstroms, S dimensionless (ranges from 0 to Zeff)
_Q_TABLE = np.array([
    0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0,
    3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 50.0, 100.0,
])
_S_TABLE = np.array([
    0.0, 0.001, 0.03, 0.12, 0.5, 3.0, 10.0, 22.0,
    32.0, 45.0, 55.0, 58.0, 62.0, 63.5, 64.8, 65.0,
])

# Build interpolator (cubic, clamped to [0, Zeff])
_s_interp = interp1d(
    _Q_TABLE, _S_TABLE,
    kind="cubic", bounds_error=False,
    fill_value=(0.0, 65.0),
)


def s_incoherent(q: torch.Tensor) -> torch.Tensor:
    """Evaluate S(q, Z=65) via interpolation. Input/output on same device."""
    q_np = q.detach().cpu().numpy()
    s_np = _s_interp(q_np).astype(np.float32)
    return torch.from_numpy(s_np).to(q.device)


def compton_energy(E0: torch.Tensor, cos_theta: torch.Tensor) -> torch.Tensor:
    """
    Compton scattered photon energy (Klein-Nishina formula).

    E' = E0 / (1 + (E0 / m_e_c2) * (1 - cos_theta))

    Args:
        E0: incident energy in MeV, (N,)
        cos_theta: cosine of scattering angle, (N,)

    Returns:
        E_prime: scattered energy in MeV, (N,)
    """
    return E0 / (1.0 + (E0 / M_E_C2_MEV) * (1.0 - cos_theta))


def compute_momentum_transfer(E_keV: torch.Tensor, cos_theta: torch.Tensor) -> torch.Tensor:
    """
    Compute momentum transfer q = sin(theta/2) / lambda.

    lambda = 12.398 / E_keV  (Angstroms)
    sin(theta/2) = sqrt((1 - cos_theta) / 2)

    Returns:
        q in inverse Angstroms
    """
    sin_half = torch.sqrt(((1.0 - cos_theta) / 2.0).clamp(min=0.0))
    wavelength = 12.398 / E_keV.clamp(min=1.0)  # Angstroms
    return sin_half / wavelength


def pixel_distance_mm(idx_a: torch.Tensor, idx_b: torch.Tensor, pitch: float = PITCH_MM) -> torch.Tensor:
    """
    Euclidean distance in mm between two pixel indices (0-24) on a 5x5 grid.

    Args:
        idx_a, idx_b: pixel indices (N,)
    Returns:
        distance in mm (N,)
    """
    ra, ca = idx_a // 5, idx_a % 5
    rb, cb = idx_b // 5, idx_b % 5
    dy = (rb.float() - ra.float()) * pitch
    dx = (cb.float() - ca.float()) * pitch
    return torch.sqrt(dx ** 2 + dy ** 2)


def estimate_scatter_angle(dist_mm: torch.Tensor, depth_mm: float = CRYSTAL_DEPTH_MM) -> torch.Tensor:
    """
    Estimate scattering angle from lateral displacement on the detector.
    Assumes photon enters normal to the detector face.

    theta ~ atan(lateral_distance / crystal_depth)
    cos_theta = depth / sqrt(depth^2 + dist^2)
    """
    return depth_mm / torch.sqrt(depth_mm ** 2 + dist_mm ** 2)


class PhysicsLoss(nn.Module):
    """
    Klein-Nishina physics constraint with S(q, Zeff) correction for LYSO.

    For each event predicted as ICS by the student:
      1. Identify predicted primary pixel and scatter pixels
      2. Estimate scattering angle from pixel positions
      3. Compute expected scattered energy via KN formula
      4. Weight by S(q, Zeff) to suppress unphysical small-angle events
      5. MSE between predicted and KN-expected energy

    This loss sends physically meaningful gradients to the student,
    encouraging it to respect Compton kinematics.
    """

    def __init__(self, ics_threshold: float = 0.5, pitch_mm: float = PITCH_MM,
                 depth_mm: float = CRYSTAL_DEPTH_MM):
        super().__init__()
        self.ics_thr = ics_threshold
        self.pitch = pitch_mm
        self.depth = depth_mm

    def forward(self, student_out: dict, x: torch.Tensor, targets: dict) -> torch.Tensor:
        """
        Args:
            student_out: dict with S_logits, P_logits, ics_logit
            x: (B, 3, 5, 5) input — channel 0 is energy
            targets: dict with is_ics, scatter_pixels, primary_pixels

        Returns:
            Scalar physics loss (0 if no ICS events in batch)
        """
        device = x.device
        B = x.shape[0]
        E_grid = x[:, 0]  # (B, 5, 5) energy map

        # Student predictions
        ics_prob = torch.sigmoid(student_out["ics_logit"])  # (B,)
        S_prob = torch.sigmoid(student_out["S_logits"]).squeeze(1)  # (B, 5, 5)
        P_prob = torch.sigmoid(student_out["P_logits"]).squeeze(1)  # (B, 5, 5)

        losses = []

        for b in range(B):
            # Only apply physics loss on ground-truth ICS events
            if not targets["is_ics"][b]:
                continue

            gt_primary = targets["primary_pixels"][b]  # list or tensor
            gt_scatter = targets["scatter_pixels"][b]

            if len(gt_primary) == 0 or len(gt_scatter) == 0:
                continue

            # Use ground-truth primary/scatter positions for physics constraint
            prim_idx = gt_primary[0] if isinstance(gt_primary, list) else gt_primary[0].item()
            prim_r, prim_c = prim_idx // 5, prim_idx % 5

            # Total energy in the event (proxy for E0 of incident photon)
            E_total = E_grid[b].sum().clamp(min=1e-6)

            for sc in gt_scatter:
                sc_idx = sc if isinstance(sc, int) else sc.item()
                sc_r, sc_c = sc_idx // 5, sc_idx % 5

                if sc_idx == prim_idx:
                    continue  # skip if primary == scatter

                # Energy at scatter pixel
                E_scatter = E_grid[b, sc_r, sc_c]
                if E_scatter < 1e-6:
                    continue

                # Lateral distance primary -> scatter
                dy = (sc_r - prim_r) * self.pitch
                dx = (sc_c - prim_c) * self.pitch
                dist = (dx ** 2 + dy ** 2) ** 0.5

                if dist < 1e-6:
                    continue

                # Estimate cos(theta)
                cos_theta = torch.tensor(
                    self.depth / (self.depth ** 2 + dist ** 2) ** 0.5,
                    device=device, dtype=torch.float32,
                )

                # KN expected scattered energy
                E_kn = compton_energy(E_total, cos_theta)

                # S(q, Zeff) correction weight
                E_keV = E_total * 1000.0  # MeV -> keV
                q = compute_momentum_transfer(E_keV, cos_theta)
                s_weight = s_incoherent(q)

                # Normalize weight to [0, 1] range (S max = Zeff ~ 65)
                s_weight_norm = (s_weight / 65.0).clamp(0.0, 1.0)

                # Soft-gate by student's scatter probability at this pixel
                s_gate = S_prob[b, sc_r, sc_c]

                # Weighted MSE: student should predict energy consistent with KN
                loss_pixel = s_weight_norm * s_gate * (E_scatter - E_kn) ** 2
                losses.append(loss_pixel)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


# =============================================================================
# Corrected physics: dσ/dΩ = dσ/dΩ_KN × S(x, Zeff)
# Reference: Hubbell (1975), NIST convention for momentum transfer x.
# =============================================================================

def klein_nishina(E_keV: torch.Tensor, theta_rad: torch.Tensor) -> torch.Tensor:
    """
    Klein-Nishina differential cross section (r_e² dropped, dimensionless ratio).

    dσ/dΩ ∝ (E'/E)² × (E'/E + E/E' − sin²θ) / 2
    where E' = E / (1 + α(1 − cosθ)),  α = E / m_e c²

    Args:
        E_keV    : incident photon energy in keV, any shape
        theta_rad: scattering angle in radians, same shape
    Returns:
        KN relative cross section (dimensionless), same shape as inputs
    """
    alpha = E_keV / 510.999
    cos_t = torch.cos(theta_rad)
    inv_factor = (1.0 + alpha * (1.0 - cos_t)).clamp(min=1e-8)
    factor = 1.0 / inv_factor          # E'/E
    kn = factor**2 * (factor + 1.0 / factor.clamp(min=1e-8) - torch.sin(theta_rad)**2)
    return kn / 2.0


def momentum_transfer(E_keV: torch.Tensor, theta_rad: torch.Tensor) -> torch.Tensor:
    """
    Momentum transfer x (NIST convention, inverse Angstroms).

    x = (E_keV / 12.398) × sin(θ / 2)

    Args:
        E_keV    : photon energy in keV, any shape
        theta_rad: scattering angle in radians, same shape
    Returns:
        x in Å⁻¹, same shape
    """
    return (E_keV / 12.398) * torch.sin(theta_rad / 2.0)


def compute_zeff(Z_list, atomic_fractions, power=2.94):
    """
    Effective atomic number Zeff = (Σ fᵢ Zᵢᵖ)^(1/p).

    Args:
        Z_list           : list of atomic numbers, e.g. [71, 64, 14, 8]
        atomic_fractions : list of atomic fractions (normalised internally)
        power            : exponent — 3.5 photoelectric, 2.94 general,
                           2.0 Compton-dominated regime
    Returns:
        Zeff as a Python float
    """
    fracs = np.array(atomic_fractions, dtype=np.float64)
    fracs /= fracs.sum()
    Zs    = np.array(Z_list, dtype=np.float64)
    return float((fracs * Zs**power).sum() ** (1.0 / power))


def S_incoherent(x: torch.Tensor, Zeff) -> torch.Tensor:
    """
    Incoherent scattering function S(x, Zeff) — Hubbell analytical form.

    S(x, Zeff) = Zeff × (1 − exp(−3 q² (1 + 1.2 q²)))
    where q = x / Zeff^(1/3)

    Values are in [0, Zeff].

    Args:
        x   : momentum transfer in Å⁻¹ (from momentum_transfer), any shape
        Zeff: effective atomic number — float scalar or tensor same shape as x
    Returns:
        S in [0, Zeff], same shape as x
    """
    if isinstance(Zeff, torch.Tensor):
        zeff_v = Zeff.to(x.device).clamp(min=1e-8)
        q = x / zeff_v ** (1.0 / 3.0)
        S = zeff_v * (1.0 - torch.exp(-3.0 * q**2 * (1.0 + 1.2 * q**2)))
        return S.clamp(min=0.0)
    else:
        zeff_f = max(float(Zeff), 1e-8)
        q = x / (zeff_f ** (1.0 / 3.0))
        S = zeff_f * (1.0 - torch.exp(-3.0 * q**2 * (1.0 + 1.2 * q**2)))
        return S.clamp(min=0.0, max=zeff_f)


def Zeff_energy_dependent(
    E_keV: torch.Tensor,
    Z_list=None,
    atomic_fractions=None,
) -> torch.Tensor:
    """
    Energy-dependent Zeff for LGSO (Lu0.62Gd1.38SiO5).

    Smoothly interpolates between photoelectric regime (power=3.5, low E)
    and Compton regime (power=2.0, high E) via a sigmoid centred at 150 keV.

    Args:
        E_keV           : photon energy in keV, any shape
        Z_list          : atomic numbers (default: LGSO_Z_LIST)
        atomic_fractions: atomic fractions (default: LGSO_FRACS)
    Returns:
        Zeff tensor, same shape as E_keV
    """
    if Z_list is None:
        Z_list = LGSO_Z_LIST
    if atomic_fractions is None:
        atomic_fractions = LGSO_FRACS
    Zeff_photo   = compute_zeff(Z_list, atomic_fractions, power=3.5)
    Zeff_compton = compute_zeff(Z_list, atomic_fractions, power=2.0)
    w = torch.sigmoid((E_keV - 150.0) / 30.0)
    return (1.0 - w) * Zeff_photo + w * Zeff_compton


def physics_loss_corrected(
    pred: torch.Tensor,
    energy_keV: torch.Tensor,
    theta_rad: torch.Tensor,
    Zeff=None,
    Z_list=None,
    atomic_fractions=None,
) -> torch.Tensor:
    """
    Physics loss with corrected incoherent scattering cross section.

    dσ/dΩ_corrected = dσ/dΩ_KN × S(x, Zeff) / Z_norm

    The student's scatter probability at each pixel is compared (MSE) against
    the normalised corrected cross section, encouraging physically consistent
    scatter predictions.

    Args:
        pred            : scatter probabilities sigmoid(S_logits), shape (N,)
        energy_keV      : incident photon energy in keV, shape (N,)
        theta_rad       : scattering angle in radians, shape (N,)
        Zeff            : float or None. None → energy-dependent Zeff.
        Z_list          : atomic numbers (default: LGSO_Z_LIST)
        atomic_fractions: atomic fractions (default: LGSO_FRACS)
    Returns:
        Scalar MSE loss
    """
    if Z_list is None:
        Z_list = LGSO_Z_LIST
    if atomic_fractions is None:
        atomic_fractions = LGSO_FRACS

    kn = klein_nishina(energy_keV, theta_rad)
    x  = momentum_transfer(energy_keV, theta_rad)

    if Zeff is None:
        zeff_t = Zeff_energy_dependent(energy_keV, Z_list, atomic_fractions)
    else:
        zeff_t = float(Zeff)

    S = S_incoherent(x, zeff_t)
    # Normalise by same Zeff used in S_incoherent: S ≤ Zeff → kn_corrected ≤ kn always
    kn_corrected = (kn * S / zeff_t).detach()   # target: no gradient through physics

    return F.mse_loss(pred, kn_corrected)


class PhysicsLossCorreected(nn.Module):
    """
    Drop-in replacement for PhysicsLoss using the corrected cross section.

    Same forward interface: forward(student_out, x_input, targets)

    Internally calls physics_loss_corrected() which uses:
        dσ/dΩ ∝ KN(E, θ) × S(x, Zeff(E)) / Z_norm
    with energy-dependent Zeff for LGSO (from GateMaterials.db).

    WARNING: This loss uses MSE(scatter_prob, KN_target) which conflicts
    with the binary scatter BCE/Focal loss. Consider PhysicsRankingLoss
    as a coherent alternative.
    """

    def __init__(
        self,
        pitch_mm: float = PITCH_MM,
        depth_mm: float = CRYSTAL_DEPTH_MM,
        Z_list=None,
        atomic_fractions=None,
    ):
        super().__init__()
        self.pitch  = pitch_mm
        self.depth  = depth_mm
        self.Z_list = Z_list or LGSO_Z_LIST
        self.fracs  = atomic_fractions or LGSO_FRACS

    def forward(self, student_out: dict, x: torch.Tensor, targets: dict) -> torch.Tensor:
        device  = x.device
        E_grid  = x[:, 0]                                              # (B,5,5) in MeV
        S_prob  = torch.sigmoid(student_out["S_logits"]).squeeze(1)   # (B,5,5)

        preds_acc  = []
        E_keV_acc  = []
        theta_acc  = []

        for b in range(x.shape[0]):
            if not targets["is_ics"][b]:
                continue

            gt_primary = targets["primary_pixels"][b]
            gt_scatter = targets["scatter_pixels"][b]
            if len(gt_primary) == 0 or len(gt_scatter) == 0:
                continue

            prim_idx = gt_primary[0] if isinstance(gt_primary, list) else gt_primary[0].item()
            prim_r, prim_c = prim_idx // 5, prim_idx % 5

            E_total_keV = E_grid[b].sum().clamp(min=1e-6) * 1000.0  # MeV → keV, tensor

            for sc in gt_scatter:
                sc_idx = sc if isinstance(sc, int) else sc.item()
                if sc_idx == prim_idx:
                    continue
                sc_r, sc_c = sc_idx // 5, sc_idx % 5
                if E_grid[b, sc_r, sc_c] < 1e-6:
                    continue

                dy = (sc_r - prim_r) * self.pitch
                dx = (sc_c - prim_c) * self.pitch
                dist = (dx**2 + dy**2) ** 0.5
                if dist < 1e-6:
                    continue

                cos_t   = self.depth / (self.depth**2 + dist**2) ** 0.5
                theta_f = math.acos(min(max(cos_t, -1.0), 1.0))
                theta_t = torch.tensor(theta_f, device=device, dtype=torch.float32)

                preds_acc.append(S_prob[b, sc_r, sc_c])
                E_keV_acc.append(E_total_keV)
                theta_acc.append(theta_t)

        if len(preds_acc) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        preds_t  = torch.stack(preds_acc)
        E_keV_t  = torch.stack(E_keV_acc)
        theta_t  = torch.stack(theta_acc)

        # physics loss corrected with S(q, Zeff) incoherent scattering function
        return physics_loss_corrected(
            preds_t, E_keV_t, theta_t,
            Zeff=None, Z_list=self.Z_list, atomic_fractions=self.fracs,
        )


class PhysicsRankingLoss(nn.Module):
    """
    Physics-coherent ranking regularizer for scatter predictions.

    Instead of MSE(scatter_prob, KN_value) — which conflicts with the binary
    scatter BCE by forcing probabilities to fractional KN values — this loss
    uses cosine similarity between the physics angular map and the student's
    scatter logits.

    For each ICS event with primary at (r_p, c_p):
      1. Pre-computed angle grid → θ for each of 25 pixels
      2. KN(E, θ) × S(x, Zeff) → physics map (5×5)
      3. Student scatter logits → S_logits (5×5)
      4. Loss = 1 − cos_sim(physics_map, S_logits)

    This constrains the spatial PATTERN (which pixels rank higher) without
    imposing absolute probability values, so it reinforces the task loss
    rather than fighting it.
    """

    def __init__(
        self,
        pitch_mm: float = PITCH_MM,
        depth_mm: float = CRYSTAL_DEPTH_MM,
        Z_list=None,
        atomic_fractions=None,
    ):
        super().__init__()
        self.Z_list = Z_list or LGSO_Z_LIST
        self.fracs  = atomic_fractions or LGSO_FRACS

        angles = torch.zeros(25, 25)
        for p in range(25):
            pr, pc = p // 5, p % 5
            for s in range(25):
                sr, sc = s // 5, s % 5
                if p == s:
                    continue
                dy = (sr - pr) * pitch_mm
                dx = (sc - pc) * pitch_mm
                dist = (dx**2 + dy**2) ** 0.5
                cos_t = depth_mm / (depth_mm**2 + dist**2) ** 0.5
                angles[p, s] = math.acos(min(max(float(cos_t), -1.0), 1.0))
        self.register_buffer("angle_grid", angles)

    def _physics_map(self, E_keV: torch.Tensor, prim_idx: int) -> torch.Tensor:
        """KN × S(x, Zeff) map for all 25 pixels given primary at prim_idx."""
        thetas = self.angle_grid[prim_idx].to(E_keV.device)     # (25,)
        kn = klein_nishina(E_keV.expand(25), thetas)             # (25,)
        x  = momentum_transfer(E_keV.expand(25), thetas)         # (25,)
        zeff = Zeff_energy_dependent(
            E_keV.expand(25), self.Z_list, self.fracs,
        )
        S = S_incoherent(x, zeff)                                # (25,)
        kn_corrected = kn * S / zeff.clamp(min=1e-8)
        kn_corrected[prim_idx] = 0.0
        return kn_corrected.view(5, 5)

    def forward(self, student_out: dict, x: torch.Tensor, targets: dict) -> torch.Tensor:
        device = x.device
        E_grid = x[:, 0]                                        # (B, 5, 5)
        S_logits = student_out["S_logits"].squeeze(1)            # (B, 5, 5)

        losses = []
        for b in range(x.shape[0]):
            if not targets["is_ics"][b]:
                continue
            gt_primary = targets["primary_pixels"][b]
            if len(gt_primary) == 0:
                continue
            prim_idx = gt_primary[0] if isinstance(gt_primary, list) else gt_primary[0].item()

            E_total_keV = E_grid[b].sum().clamp(min=1e-6) * 1000.0

            phys_map = self._physics_map(E_total_keV, int(prim_idx))
            phys_flat = phys_map.flatten()
            s_flat = S_logits[b].flatten()

            if phys_flat.norm() < 1e-8:
                continue

            cos_sim = F.cosine_similarity(
                phys_flat.unsqueeze(0), s_flat.unsqueeze(0),
            )
            losses.append(1.0 - cos_sim.squeeze())

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)
        return torch.stack(losses).mean()
