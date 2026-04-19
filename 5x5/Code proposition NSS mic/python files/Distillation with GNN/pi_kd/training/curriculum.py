"""
Curriculum scheduler for progressive loss activation + temperature annealing.

Supports two modes controlled by ``num_phases``:

  2-phase (default):
    Phase 1 (0 .. task_frac):   L_task only
    Phase 2 (task_frac .. end): L_task + L_KD + L_phys  (all ramp together)

  3-phase (legacy / ablation):
    Phase 1 (0 .. p1_end):      L_task only
    Phase 2 (p1_end .. p2_end): L_task + L_KD
    Phase 3 (p2_end .. end):    L_task + L_KD + L_phys

Temperature annealing (cosine):
  T decays from T_start to T_end over all KD-active epochs.

Ablation study usage:
  --no-teacher  => lambda_kd=0 (task + physics only)
  --no-physics  => lambda_phys=0 (task + KD only)
  --no-teacher --no-physics => pure task baseline
"""

import math


class CurriculumScheduler:
    """
    Manages lambda_kd, lambda_phys, lambda_feat, and temperature over epochs.

    Args:
        total_epochs: total training epochs
        num_phases: 2 or 3 (see module docstring)
        phase1_frac: fraction of epochs for task-only phase
        phase2_frac: (3-phase only) fraction for task+KD phase
        lambda_kd_max, lambda_phys_max, lambda_feat_max: peak weights
        ramp_epochs: linear ramp-up length within each phase
        T_start, T_end: temperature annealing bounds
    """

    def __init__(
        self,
        total_epochs: int = 90,
        num_phases: int = 2,
        phase1_frac: float = 0.50,
        phase2_frac: float = 0.50,
        lambda_kd_max: float = 1.0,
        lambda_phys_max: float = 0.1,
        lambda_feat_max: float = 0.0,
        ramp_epochs: int = 3,
        T_start: float = 2.0,
        T_end: float = 1.2,
    ):
        self.total_epochs = total_epochs
        self.num_phases = num_phases
        self.lambda_kd_max = lambda_kd_max
        self.lambda_phys_max = lambda_phys_max
        self.lambda_feat_max = lambda_feat_max
        self.ramp_epochs = max(ramp_epochs, 1)
        self.T_start = T_start
        self.T_end = T_end

        self.phase1_end = int(total_epochs * phase1_frac)

        if num_phases == 3:
            self.phase2_end = int(total_epochs * (phase1_frac + phase2_frac))
        else:
            self.phase2_end = total_epochs

    def _temperature(self, epoch: int) -> float:
        """Cosine annealing of T from T_start to T_end over KD-active epochs."""
        if epoch < self.phase1_end:
            return self.T_start
        kd_epochs = self.total_epochs - self.phase1_end
        if kd_epochs <= 0:
            return self.T_end
        progress = min(1.0, (epoch - self.phase1_end) / kd_epochs)
        cos_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.T_end + (self.T_start - self.T_end) * cos_factor

    def get_lambdas(self, epoch: int) -> dict:
        temperature = self._temperature(epoch)

        # Phase 1: task only (both 2-phase and 3-phase)
        if epoch < self.phase1_end:
            return {
                "lambda_kd": 0.0, "lambda_phys": 0.0, "lambda_feat": 0.0,
                "temperature": temperature,
            }

        if self.num_phases == 2:
            # Phase 2: everything ramps together
            epochs_in = epoch - self.phase1_end
            ramp = min(1.0, epochs_in / self.ramp_epochs)
            return {
                "lambda_kd": self.lambda_kd_max * ramp,
                "lambda_phys": self.lambda_phys_max * ramp,
                "lambda_feat": self.lambda_feat_max * ramp,
                "temperature": temperature,
            }

        # --- 3-phase legacy ---
        if epoch < self.phase2_end:
            epochs_in = epoch - self.phase1_end
            ramp = min(1.0, epochs_in / self.ramp_epochs)
            return {
                "lambda_kd": self.lambda_kd_max * ramp,
                "lambda_phys": 0.0,
                "lambda_feat": self.lambda_feat_max * ramp,
                "temperature": temperature,
            }

        epochs_in = epoch - self.phase2_end
        ramp = min(1.0, epochs_in / self.ramp_epochs)
        return {
            "lambda_kd": self.lambda_kd_max,
            "lambda_phys": self.lambda_phys_max * ramp,
            "lambda_feat": self.lambda_feat_max,
            "temperature": temperature,
        }

    def get_phase_name(self, epoch: int) -> str:
        if epoch < self.phase1_end:
            return "Phase1:Task"

        if self.num_phases == 2:
            components = []
            if self.lambda_kd_max > 0:
                components.append("KD")
            if self.lambda_phys_max > 0:
                components.append("Phys")
            if self.lambda_feat_max > 0:
                components.append("Feat")
            suffix = "+".join(components) if components else "Aux"
            return f"Phase2:Task+{suffix}"

        if epoch < self.phase2_end:
            return "Phase2:Task+KD"
        return "Phase3:Task+KD+Phys"

    def phase_boundaries(self) -> list[int]:
        """Return epoch indices where phase transitions occur."""
        if self.num_phases == 2:
            return [self.phase1_end]
        return [self.phase1_end, self.phase2_end]

    def __repr__(self) -> str:
        if self.num_phases == 2:
            return (
                f"CurriculumScheduler(2-phase=[0-{self.phase1_end}:Task, "
                f"{self.phase1_end}-{self.total_epochs}:+KD+Phys], "
                f"kd={self.lambda_kd_max}, phys={self.lambda_phys_max}, "
                f"feat={self.lambda_feat_max}, T={self.T_start}->{self.T_end})"
            )
        return (
            f"CurriculumScheduler(3-phase=[0-{self.phase1_end}:Task, "
            f"{self.phase1_end}-{self.phase2_end}:+KD, "
            f"{self.phase2_end}-{self.total_epochs}:+Phys], "
            f"kd={self.lambda_kd_max}, phys={self.lambda_phys_max}, "
            f"feat={self.lambda_feat_max}, T={self.T_start}->{self.T_end})"
        )
