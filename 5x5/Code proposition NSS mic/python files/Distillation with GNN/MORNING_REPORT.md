# MORNING REPORT — Reduction widths accumulateurs HLS

**Date run** : lancé à `20260820_012236`
**Sujet** : StudentNet small_v2 (2338 params INT8), ASIC SKY130 HD
**Objectif** : réduire les registres (baseline 389 601 DFFs, ~64% de l'aire) en coupant les bits fractionnaires superflus des accumulateurs `ap_fixed`.

---

## 1. Modification appliquée (un seul fichier)

**Fichier** : `/data/jdil1901/hls_work_small/firmware/defines.h`
**Backup baseline** : `defines.h.baseline_20260820_012236`

| Layer | accum_t baseline | accum_t nouveau | Delta bits |
|---|---|---|---:|
| DW Conv_0 (3ch, block1)    | `ap_fixed<21,13>` | `ap_fixed<17,13>` | −4 (frac) |
| PW Conv_1 (3→16)           | `ap_fixed<19,11>` | `ap_fixed<15,11>` | −4 (frac) |
| DW Conv_2 (16ch, block2)   | `ap_fixed<21,13>` | `ap_fixed<17,13>` | −4 (frac) |
| PW Conv_3 (16→32)          | `ap_fixed<21,13>` | `ap_fixed<17,13>` | −4 (frac) |
| DW Conv_4 (32ch, block3)   | `ap_fixed<21,13>` | `ap_fixed<17,13>` | −4 (frac) |
| PW Conv_5 (32→32)          | `ap_fixed<22,14>` | `ap_fixed<18,14>` | −4 (frac) |
| PW Conv_6 (head, 32→1)     | `ap_fixed<22,14>` | `ap_fixed<18,14>` | −4 (frac) |

**Bits integer conservés** → aucun risque de saturation (même bornes que le baseline).
**Bits fractionnaires** : 8-10 → 4 → suffit car la sortie de chaque Conv est tronquée à `ap_fixed<8,4>` (4 frac).
Erreur d'accumulation bornée à `fan_in × 2⁻⁴ / 2 ≈ 0.3 LSB output` (largement < 1 LSB).

---

## 2. Résultats Vitis HLS (csynth)

| Métrique | Baseline (auto) | Réduit (17/18) | Δ |
|---|---:|---:|---:|
| Clock achieved (ns) | ? | 4.373 | n/a |
| Latency max (cycles) | ? | 762 | n/a |
| II max (cycles) | ? | 200 | n/a |
| FF (total) | ? | 854,814 | n/a |
| LUT (total) | ? | 854,472 | n/a |
| FF - FIFO | ? | 648,450 | n/a |
| FF - Instance (convs) | ? | 199,814 | n/a |

---

## 3. Résultats Yosys + SKY130

| Métrique | Baseline | Réduit | Δ |
|---|---:|---:|---:|
| Aire chip (µm²) | 314,951 | 323,124 | +2.6% |
| Gate equivalent (kGE) | 0.720 | 0.330 | -54.2% |
| Critical-path delay (ps) | 313.910 | 313.910 | +0.0% |
| Fmax ABC (MHz) | 3,186 | 3,186 | +0.0% |
| Total cellules | 1,719,549 | 1,662,042 | -3.3% |
| Aire cumulée cellules (µm²) | 16,600,000 | 16,300,000 | -1.8% |

## 4. Résultats DFFs (compte netlist gate-level)

| Type de DFF | Baseline | Réduit | Δ |
|---|---:|---:|---:|
| dfxtp_1 (simple DFF) | 82,521 | 84,569 | +2.5% |
| edfxtp_1 (enable DFF) | 130,520 | 117,653 | -9.9% |
| TOTAL DFFs | 213,041 | 202,222 | -5.1% |

---

## 5. Fichiers

### Nouveau run
- Vitis HLS csynth: `/data/jdil1901/hls_work_small/myproject_prj/solution1/syn/report/myproject_csynth.rpt`
- Yosys report: `/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/asic_sky130/work/report.txt`
- Netlist gate-level: `/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/asic_sky130/work/synth_netlist.v`
- Cell breakdown: `/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/asic_sky130/work/cell_breakdown.txt`
- defines.h utilisé: `/data/jdil1901/hls_work_small/firmware/defines.h`

### Baseline (avant modif)
- Vitis HLS csynth baseline: `/data/jdil1901/hls_work_small/myproject_prj/solution1/syn/report/myproject_csynth.rpt.baseline_20260820_012236`
- Yosys report baseline: `/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/asic_sky130/work/report.txt.baseline_20260820_012236`
- Netlist baseline: `/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/asic_sky130/work/synth_netlist.v.baseline_20260820_012236`
- Cell breakdown baseline: `/data/jdil1901/Documents/ics/code/5x5/Code proposition NSS mic/python files/Distillation with GNN/asic_sky130/work/cell_breakdown.txt.baseline_20260820_012236`
- defines.h baseline: `/data/jdil1901/hls_work_small/firmware/defines.h.baseline_20260820_012236`

---

## 6. Rollback (si besoin)

Pour revenir à la version baseline en 2 commandes :

```bash
cp /data/jdil1901/hls_work_small/firmware/defines.h.baseline_20260820_012236 /data/jdil1901/hls_work_small/firmware/defines.h
cd /data/jdil1901/hls_work_small && vitis_hls -f build_prj.tcl reset=1 csim=0 synth=1 cosim=0 validation=0 export=0
```

---

## 7. Prochaines étapes suggérées

1. **Vérifier AUPRC/AccTop1** en re-lançant le testbench Vitis HLS avec les nouvelles widths si les inputs de test existent — sinon vérifier bit-accuracy en comparant les sorties C-sim baseline vs nouvelles pour quelques inputs types.
2. Si tout est OK → mettre à jour la mémoire projet avec les nouveaux chiffres pour l'abstract NSS.
3. Si dégradation visible → repasser à 6 bits frac (accum `<19,13>` etc.), toujours sans re-training.

---
*Rapport généré automatiquement — check `bg_overnight_widths.log` pour les logs complets.*