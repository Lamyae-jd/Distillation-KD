"""
Parse a PI-KD training log and generate figures.

Can be used as:
  - Importable: from pi_kd.scripts.plot_training import generate_figures
  - Standalone:  python -m pi_kd.scripts.plot_training logs/run_XXXX.log
"""

import re
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPOCH_PATTERN = re.compile(
    r"Ep\s+(\d+)\s+\[(\S+)\s*\]\s+T=\s*([\d.]+)\s+"
    r"loss=([\d.]+)\s+\(task=([\d.]+)\s+kd=([\d.]+)\s+feat=([\d.]+)\s+phys=([\d.]+)\)\s+\|\s+"
    r"S_F1=([\d.]+)\(ap=([\d.]+)\)\s+"
    r"P_F1=([\d.]+)\(ap=([\d.]+)\)\s+"
    r"ICS_F1=([\d.]+)\(ap=([\d.]+)\)\s+"
    r"score=([\d.]+)\s*(\*)?\s+"
    r"lr=([\d.e+-]+)"
)

COLORS = {
    "primary": "#2563eb", "scatter": "#059669", "ics": "#dc2626",
    "kd": "#7c3aed", "phys": "#ea580c", "task": "#2563eb",
    "total": "#1e293b", "temp": "#9333ea", "lr": "#64748b",
}


def parse_log(log_path):
    data = {
        "epochs": [], "phases": [], "temps": [], "lrs": [], "scores": [],
        "total_loss": [], "task_loss": [], "kd_loss": [], "phys_loss": [],
        "s_f1": [], "s_ap": [], "p_f1": [], "p_ap": [],
        "ics_f1": [], "ics_ap": [],
    }
    with open(log_path) as f:
        for line in f:
            m = EPOCH_PATTERN.search(line)
            if not m:
                continue
            data["epochs"].append(int(m.group(1)))
            data["phases"].append(m.group(2))
            data["temps"].append(float(m.group(3)))
            data["total_loss"].append(float(m.group(4)))
            data["task_loss"].append(float(m.group(5)))
            data["kd_loss"].append(float(m.group(6)))
            data["phys_loss"].append(float(m.group(8)))
            data["s_f1"].append(float(m.group(9)))
            data["s_ap"].append(float(m.group(10)))
            data["p_f1"].append(float(m.group(11)))
            data["p_ap"].append(float(m.group(12)))
            data["ics_f1"].append(float(m.group(13)))
            data["ics_ap"].append(float(m.group(14)))
            data["scores"].append(float(m.group(15)))
            data["lrs"].append(float(m.group(17)))

    for k in data:
        if k != "phases":
            data[k] = np.array(data[k])
    return data


def _add_phase_shading(ax, epochs, p1_end, p2_end, ymin=None, ymax=None):
    if ymin is None:
        ymin, ymax = ax.get_ylim()
    ax.axvspan(epochs[0], p1_end + 0.5, alpha=0.06, color="#3b82f6", zorder=0)
    ax.axvspan(p1_end + 0.5, p2_end + 0.5, alpha=0.06, color="#8b5cf6", zorder=0)
    ax.axvspan(p2_end + 0.5, epochs[-1], alpha=0.06, color="#f97316", zorder=0)
    ax.text((epochs[0] + p1_end) / 2, ymax * 0.97, "Phase 1\nTask only",
            ha="center", va="top", fontsize=7, color="#3b82f6", alpha=0.7)
    ax.text((p1_end + p2_end) / 2 + 0.5, ymax * 0.97, "Phase 2\nTask + KD",
            ha="center", va="top", fontsize=7, color="#7c3aed", alpha=0.7)
    ax.text((p2_end + epochs[-1]) / 2 + 0.5, ymax * 0.97, "Phase 3\nTask+KD+Phys",
            ha="center", va="top", fontsize=7, color="#ea580c", alpha=0.7)


def generate_figures(log_path, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    d = parse_log(log_path)

    if len(d["epochs"]) == 0:
        print(f"WARNING: no epoch data found in {log_path}")
        return

    epochs = d["epochs"]
    phases = d["phases"]

    has_p1 = any("Phase1" in p for p in phases)
    has_p2 = any("Phase2" in p for p in phases)
    has_p3 = any("Phase3" in p for p in phases)

    p1_end = max((e for e, p in zip(epochs, phases) if "Phase1" in p), default=epochs[0])
    p2_end = max((e for e, p in zip(epochs, phases) if "Phase2" in p), default=p1_end)
    shade = lambda ax, **kw: _add_phase_shading(ax, epochs, p1_end, p2_end, **kw)

    # ── Figure 1: AUPRC & F1 ────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(epochs, d["p_ap"], "-o", color=COLORS["primary"], markersize=3, linewidth=1.5, label="Primary AUPRC (= score)")
    ax1.plot(epochs, d["s_ap"], "-s", color=COLORS["scatter"], markersize=3, linewidth=1.5, label="Scatter AUPRC")
    ax1.plot(epochs, d["ics_ap"], "-^", color=COLORS["ics"], markersize=3, linewidth=1.5, label="ICS AUPRC")
    best_idx = np.argmax(d["scores"])
    ax1.axhline(d["scores"][best_idx], color=COLORS["primary"], linestyle="--", alpha=0.4, linewidth=0.8)
    ax1.annotate(f"Best: {d['scores'][best_idx]:.4f} (ep {epochs[best_idx]})",
                 xy=(epochs[best_idx], d["scores"][best_idx]), fontsize=8,
                 xytext=(epochs[best_idx] - 10, d["scores"][best_idx] + 0.008),
                 arrowprops=dict(arrowstyle="->", color=COLORS["primary"], lw=0.8),
                 color=COLORS["primary"])
    ax1.set_ylabel("AUPRC")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_title("PI-KD Student Training — Validation Metrics", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    shade(ax1)

    ax2.plot(epochs, d["p_f1"], "-o", color=COLORS["primary"], markersize=3, linewidth=1.5, label="Primary F1")
    ax2.plot(epochs, d["s_f1"], "-s", color=COLORS["scatter"], markersize=3, linewidth=1.5, label="Scatter F1")
    ax2.plot(epochs, d["ics_f1"], "-^", color=COLORS["ics"], markersize=3, linewidth=1.5, label="ICS F1")
    ax2.set_ylabel("F1 Score")
    ax2.set_xlabel("Epoch")
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(True, alpha=0.2)
    shade(ax2)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_auprc_f1.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Loss breakdown ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(epochs, d["total_loss"], "-", color=COLORS["total"], linewidth=2, label="Total loss", zorder=5)
    ax.plot(epochs, d["task_loss"], "--", color=COLORS["task"], linewidth=1.2, label="Task (scaled)")
    ax.plot(epochs, d["kd_loss"], "--", color=COLORS["kd"], linewidth=1.2, label="KD (scaled)")
    ax.plot(epochs, d["phys_loss"], "--", color=COLORS["phys"], linewidth=1.2, label="Physics (scaled)")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_title("PI-KD Loss Breakdown (all values after adaptive scaling)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)
    shade(ax)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "loss_breakdown.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Temperature & LR schedule ──────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(10, 3.5))
    ax1.plot(epochs, d["temps"], "-", color=COLORS["temp"], linewidth=2, label="Temperature T")
    ax1.set_ylabel("Temperature", color=COLORS["temp"])
    ax1.tick_params(axis="y", labelcolor=COLORS["temp"])
    ax2 = ax1.twinx()
    ax2.plot(epochs, d["lrs"], "-", color=COLORS["lr"], linewidth=1.5, alpha=0.7, label="Learning rate")
    ax2.set_ylabel("Learning rate", color=COLORS["lr"])
    ax2.tick_params(axis="y", labelcolor=COLORS["lr"])
    ax2.set_yscale("log")
    ax1.set_xlabel("Epoch")
    ax1.set_title("Temperature Annealing & Learning Rate Schedule", fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.2)
    shade(ax1, ymin=min(d["temps"]) - 0.1, ymax=max(d["temps"]) + 0.2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "temperature_lr.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: Phase 2 zoom (only if Phase 2 exists) ─────────────────────
    if has_p2:
        mask_p1_late = (epochs >= p1_end - 5) & (epochs <= p1_end)
        mask_p2 = (epochs > p1_end) & (epochs <= p2_end)
        mask_zoom = mask_p1_late | mask_p2

        if mask_zoom.sum() > 2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(epochs[mask_zoom], d["p_ap"][mask_zoom], "-o", color=COLORS["primary"], markersize=5, linewidth=2, label="Primary AUPRC")
            ax.axvline(p1_end + 0.5, color="gray", linestyle=":", alpha=0.5, label="Phase 1→2 transition")
            ax.fill_between(epochs[mask_zoom], d["p_ap"][mask_zoom].min() - 0.002,
                            d["p_ap"][mask_zoom], alpha=0.1, color=COLORS["primary"])
            p1_best = d["p_ap"][mask_p1_late].max() if mask_p1_late.any() else 0
            p2_best = d["p_ap"][mask_p2].max() if mask_p2.any() else 0
            ax.axhline(p1_best, color="#94a3b8", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.axhline(p2_best, color=COLORS["primary"], linestyle="--", alpha=0.5, linewidth=0.8)
            ax.annotate(f"Phase 1 peak: {p1_best:.4f}", xy=(epochs[mask_p1_late][-1], p1_best),
                        fontsize=8, color="#64748b", va="bottom")
            ax.annotate(f"Phase 2 peak: {p2_best:.4f} (+{p2_best - p1_best:.4f})",
                        xy=(epochs[mask_p2][np.argmax(d["p_ap"][mask_p2])], p2_best),
                        fontsize=9, fontweight="bold", color=COLORS["primary"], va="bottom")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Primary AUPRC")
            ax.set_title("KD Impact: Phase 1 → Phase 2 Transition", fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.2)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "kd_impact_zoom.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ── Figure 5: Phase 3 zoom (only if Phase 3 exists) ─────────────────────
    if has_p3:
        mask_p2_late = (epochs >= p2_end - 5) & (epochs <= p2_end)
        mask_p3 = epochs > p2_end
        mask_crash = mask_p2_late | mask_p3

        if mask_crash.sum() > 2:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            ax1.plot(epochs[mask_crash], d["p_ap"][mask_crash], "-o", color=COLORS["primary"], markersize=4, linewidth=1.8, label="Primary AUPRC")
            ax1.plot(epochs[mask_crash], d["s_ap"][mask_crash], "-s", color=COLORS["scatter"], markersize=4, linewidth=1.8, label="Scatter AUPRC")
            ax1.axvline(p2_end + 0.5, color="gray", linestyle=":", alpha=0.5)
            ax1.set_ylabel("AUPRC")
            ax1.set_title("Phase 3 Impact: Physics Loss Effect", fontsize=11, fontweight="bold")
            ax1.legend(fontsize=8)
            ax1.grid(True, alpha=0.2)

            ax2.plot(epochs[mask_crash], d["phys_loss"][mask_crash], "-", color=COLORS["phys"], linewidth=2, label="Physics loss (scaled)")
            ax2.plot(epochs[mask_crash], d["task_loss"][mask_crash], "--", color=COLORS["task"], linewidth=1.5, label="Task loss")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.2)
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, "phase3_impact.png"), dpi=200, bbox_inches="tight")
            plt.close(fig)

    best_idx = np.argmax(d["scores"])
    print(f"  Figures saved to {out_dir}/ (best score: {d['scores'][best_idx]:.4f} at ep {epochs[best_idx]})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m pi_kd.scripts.plot_training <log_file> [output_dir]")
        sys.exit(1)
    log_file = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "figures"
    generate_figures(log_file, out)
