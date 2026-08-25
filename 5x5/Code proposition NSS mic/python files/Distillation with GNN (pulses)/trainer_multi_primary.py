import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from loss_multi_primary import BinaryFocalLoss
from tqdm import tqdm



# Import des losses (à adapter selon ton organisation)
from loss_multi_primary import (
    bce_hit_loss, l1_energy_loss, l1_time_loss, energy_conservation_loss,
    tof_consistency_loss, fraction_cap_loss, spatial_confinement_loss,
    ce_primary_multilabel_loss, focal_primary_loss, multilabel_ranking_loss,
    ics_score_from_P,
    primary_top1_accuracy, acc_within_radius, primary_recall_precision_f1,
    scatter_metrics, scatter_auprc_auroc, scatter_metrics_per_pulse,
    assignment_loss, correction_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Trainer:
    def __init__(self, model, optimizer, scaler, cfg, dist_mm, pos_weight_P=None, pos_weight_S=None, pos_weight_ICS=None, ckpt_dir="./checkpoints"):
        self.model = model
        self.opt = optimizer
        self.scaler = scaler
        self.cfg = cfg

        device = next(model.parameters()).device
        self.dist_mm = dist_mm.to(device)
        self.pos_weight_P   = pos_weight_P.to(device)   if pos_weight_P   is not None else None
        self.pos_weight_S   = pos_weight_S.to(device)   if pos_weight_S   is not None else None
        # ICS imbalance weight: ~(n_non_ics / n_ics). Pass from main_train or use default 10.
        self.pos_weight_ICS = pos_weight_ICS.to(device) if pos_weight_ICS is not None else \
                              torch.tensor([10.0], device=device)

        self.ckpt_dir = ckpt_dir
        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_score = -1.0

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="max", factor=0.5, patience=2
        )

        # Poids des pertes (ajustés pour multi-primaires + assignment + correction)
        self.LAM = dict(
            scatter = 0.2,
            hit     = 0.2,
            E       = 0.001,
            t       = 0.003,
            TOF     = 0.005,
            frac    = 0.1,
            conf    = 0.4,
            prim    = 4.0,
            ics     = 4.0,   # ICS gate head (pulse-level binary) — fort signal nécessaire
            assign  = 1.0,   # assignment: scatter pixel → primary linking
            corr    = 1.0,   # correction: scatter-free energy map
        )
        
        self.focal_scatter = BinaryFocalLoss(alpha=0.25, gamma=2.5, reduction="mean")  
        self.use_focal_for_scatter = True  # toggle


    def train_one_epoch(self, loader, epoch):
        from tqdm import tqdm
        self.model.train()
        warm_T_epochs = int(self.cfg.get("train", {}).get("warm_T_epochs", 0))
        mp = self.cfg["mixed_precision"]
        warm_T_epochs = self.cfg["train"]["warm_T_epochs"]
        losses = []

        use_amp = (DEVICE == "cuda") and mp
        bf16_ok = torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
        amp_dtype = torch.bfloat16 if (use_amp and bf16_ok) else torch.float16

        pbar = tqdm(loader, desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)
        for it, (x, y) in enumerate(pbar):
            x = x.to(DEVICE)
            S_star = y["S_star"].to(DEVICE).unsqueeze(1)
            P_star = y["P_star"].to(DEVICE).unsqueeze(1)
            E_star = y["E_star"].to(DEVICE).unsqueeze(1)
            T_star = y["T_star"].to(DEVICE).unsqueeze(1)
            is_ics = y["is_ics"].to(DEVICE)
            primary_pixels_list = y["primary_pixels"]  # Liste de tensors
            assign_map_gt = y["assign_map"].to(DEVICE)  # (B, 5, 5)
            E_corrected_gt = y["E_corrected"].to(DEVICE)  # (B, 5, 5)

            # Curriculum: utiliser T* comme biais pendant warm-up
            # use_T_bias = T_star if epoch <= warm_T_epochs else None
            use_T_bias = T_star
            # print(use_T_bias)
            learn_bias = (epoch > warm_T_epochs)
            
            self.opt.zero_grad(set_to_none=True)

           
            warm_T_epochs = self.cfg["train"]["warm_T_epochs"]

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                # out = self.model(x, use_T_bias=(T_star if epoch <= warm_T_epochs else None),
                #                  learn_bias=(epoch > warm_T_epochs))
                out = self.model(x, T_star, learn_bias=(epoch > warm_T_epochs))
                S_logits, S = out["S_logits"], out["S"]
                P_logits, P = out["P_logits"], out["P"]
                logitsK_raw = out["primary_logits_raw"]   # raw — for ranking loss (no ICS gradient conflict)
                idxK = out["primary_idx"]
                ics_logit = out["ics_logit"]   # (B,)
                assign_targets = out["assign_targets"]     # (B, 2, 5, 5)
                E_corr_pred = out["E_corrected"]           # (B, 1, 5, 5)

                E = out.get("E", x[:, 0:1, :, :])   # fallback: use input energy channel
                T = out.get("T", None)
            
                # main losses
                L_scatter = (self.focal_scatter(S_logits, S_star)
                             if getattr(self, "use_focal_for_scatter", False)
                             else bce_hit_loss(S_logits, S_star, pos_weight=self.pos_weight_S))
                L_hit  = bce_hit_loss(P_logits, P_star, pos_weight=self.pos_weight_P)
                L_prim = multilabel_ranking_loss(logitsK_raw, idxK, primary_pixels_list, is_ics)
                L_ics  = F.binary_cross_entropy_with_logits(
                             ics_logit, is_ics.float(),
                             pos_weight=self.pos_weight_ICS
                         )
                L_conf = spatial_confinement_loss(P, primary_pixels_list,
                                                  radius_px=self.cfg["model"]["confinement_radius_px"])
                
                P_prob_mean = P.view(P.shape[0], -1).mean()
                # min_selection_rate = 0.20
                # selection_penalty = F.relu(min_selection_rate - P_prob_mean) * 5.0  # Forte penalite
                
                # L_prim = L_prim + selection_penalty
                L_frac = fraction_cap_loss(
                    E.squeeze(1), primary_pixels_list,
                    cap=self.cfg["model"]["primary_cap"],
                    tol=self.cfg["model"]["primary_cap_tol"],
                    is_ics=is_ics
                )
            
                # --- AUX: only compute if heads exist AND after warm-up ---
                L_E = L_T = L_TOF = torch.tensor(0.0, device=x.device)
                if (E is not None) and (T is not None) and (epoch > warm_T_epochs):
                    # mask only on hits (P_star>0) and ICS pulses (optional)
                    mask_hits = (P_star > 0).float()
                    if is_ics.ndim == 1:  # (B,)
                        mask_ics = is_ics.view(-1, 1, 1, 1).float()
                    else:
                        mask_ics = is_ics.float()
                    mask = mask_hits * mask_ics
            
                    # tiny-weighted aux losses
                    L_E   = l1_energy_loss(E, E_star, mask=mask)         # tiny
                    L_T   = l1_time_loss(T, T_star, mask=mask)           # tiny
                    L_TOF = tof_consistency_loss(T.squeeze(1),
                                                 self.dist_mm,
                                                 c_mm_per_ns=self.cfg["model"]["c_mm_per_ns"],
                                                 sigma_t_ns=self.cfg["model"]["sigma_t_ns"],
                                                 P_mask=(mask.squeeze(1)>0).float())  # tiny
            
                # Assignment loss: scatter pixel → primary linking
                L_assign = assignment_loss(assign_targets, assign_map_gt,
                                           primary_pixels_list, S_star)
                # Correction loss: scatter-free energy map
                L_corr = correction_loss(E_corr_pred, E_corrected_gt, E)

                loss = (
                    self.LAM["scatter"] * L_scatter +
                    self.LAM["frac"]    * L_frac    +
                    self.LAM["hit"]     * L_hit     +
                    self.LAM["conf"]    * L_conf    +
                    self.LAM["prim"]    * L_prim    +
                    self.LAM["ics"]     * L_ics     +
                    self.LAM["assign"]  * L_assign  +
                    self.LAM["corr"]    * L_corr    +
                    self.LAM["E"]       * L_E       +
                    self.LAM["t"]       * L_T       +
                    self.LAM["TOF"]     * L_TOF
                )
            
            
           
            # Backward + step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.opt)
            self.scaler.update()

            losses.append(loss.item())
            
            pbar.set_postfix(
                loss=f"{loss.item():.3f}",
                hit=f"{L_hit.item():.2f}",
                prim=f"{L_prim.item():.2f}",
                sc=f"{L_scatter.item():.2f}",
                ics=f"{L_ics.item():.2f}",
                asgn=f"{L_assign.item():.2f}",
                corr=f"{L_corr.item():.2f}",
            )

        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def validate(self, loader):
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
        import numpy as np
        import math
        self.model.eval()
    
        # ---- Collectors ----
        y_true_ics, y_score_ics = [], []         # ICS (pulse)
        all_S_logits, all_S_star = [], []        # Scatter (pixel)
        all_logitsK, all_idxK, all_primary_pixels = [], [], []  # Primary candidates
    
        corr_mae_sum, corr_count = 0.0, 0

        for x, y in tqdm(loader, desc="Validation", unit="batches", dynamic_ncols=True, leave=False):
            x = x.to(DEVICE)
            S_star = y["S_star"].to(DEVICE).unsqueeze(1)  # (B,1,5,5)


            T_star = y["T_star"].to(DEVICE).unsqueeze(1)      # (B,1,H,W)
            out = self.model(x, use_T_bias=T_star)
            S_logits = out["S_logits"]           # (B,1,5,5)
            P_logits = out["P_logits"]
            logitsK  = out["primary_logits"]     # (B,K)
            idxK     = out["primary_idx"]        # (B,K)

            # ========= ICS via dedicated ICS head =========
            ics_prob = torch.sigmoid(out["ics_logit"])
            ics_prob = torch.nan_to_num(ics_prob, nan=0.0, posinf=1.0, neginf=0.0)
            y_score_ics.extend(ics_prob.cpu().numpy().tolist())
            y_true_ics.extend(y["is_ics"].numpy().astype(int).tolist())

            # ========= Scatter (pixel) collection =========
            all_S_logits.append(S_logits.cpu())
            all_S_star.append(S_star.cpu())

            # ========= Primary candidates collection =========
            all_logitsK.append(logitsK.cpu())
            all_idxK.append(idxK.cpu())
            all_primary_pixels.extend(y["primary_pixels"])

            # ========= Correction metrics =========
            if "E_corrected" in y:
                E_corr_pred = out["E_corrected"].cpu()           # (B, 1, 5, 5)
                E_corr_gt = y["E_corrected"].unsqueeze(1)        # (B, 1, 5, 5)
                mae = (E_corr_pred - E_corr_gt).abs().mean().item()
                corr_mae_sum += mae * x.shape[0]
                corr_count += x.shape[0]
                
        # ======= Stack scatter tensors =======
        all_S_logits = torch.cat(all_S_logits, dim=0)  # (N,1,5,5)
        all_S_star   = torch.cat(all_S_star,   dim=0)  # (N,1,5,5)
    
        # Simple diagnostic for NaN/Inf
        def _nan_report(name, t):
            t_flat = t.view(-1)
            n_nan  = torch.isnan(t_flat).sum().item()
            n_inf  = torch.isinf(t_flat).sum().item()
            if n_nan or n_inf:
                print(f"[warn] {name}: NaN={n_nan}, Inf={n_inf}")
            return n_nan, n_inf
        _nan_report("all_S_logits", all_S_logits)
    
        # ======= ICS AUPRC/BestThr + gate threshold at target recall =======
        TARGET_ICS_RECALL = 0.80   # accepter de rater 20% des ICS pour mieux filtrer les non-ICS
        ics_gate_thr = 0.5
        ics_recall_at_gate = float("nan")
        ics_prec_at_gate   = float("nan")

        if len(set(y_true_ics)) > 1:
            auprc_ics = average_precision_score(y_true_ics, y_score_ics)
            prec_ics, rec_ics, thr_ics = precision_recall_curve(y_true_ics, y_score_ics)
            f1_ics = 2*prec_ics*rec_ics/(prec_ics+rec_ics+1e-9)
            best_idx_ics = int(np.argmax(f1_ics))
            best_thr_ics = thr_ics[best_idx_ics] if best_idx_ics < len(thr_ics) else 0.5

            # Gate threshold: highest threshold where ICS recall >= TARGET
            # rec_ics[:-1] aligns with thr_ics (sklearn convention)
            valid_idx = np.where(rec_ics[:-1] >= TARGET_ICS_RECALL)[0]
            if len(valid_idx) > 0:
                gi = valid_idx[-1]   # highest threshold still meeting recall target
                ics_gate_thr       = float(thr_ics[gi])
                ics_recall_at_gate = float(rec_ics[gi])
                ics_prec_at_gate   = float(prec_ics[gi])
            else:
                # Can't reach target recall → use lowest threshold available
                ics_gate_thr       = float(thr_ics[0]) if len(thr_ics) > 0 else 0.1
                ics_recall_at_gate = float(rec_ics[0])
                ics_prec_at_gate   = float(prec_ics[0])
        else:
            auprc_ics = float("nan"); best_thr_ics = 0.5
    
        # ======= Scatter PR-based best threshold =======
        S_prob_flat = torch.sigmoid(all_S_logits).view(-1)
        S_prob_flat = torch.nan_to_num(S_prob_flat, nan=0.0, posinf=1.0, neginf=0.0)
        S_prob_flat = S_prob_flat.cpu().numpy()
        S_true_flat = all_S_star.view(-1).numpy().astype(int)
    
        if len(np.unique(S_true_flat)) > 1:
            prec_sc, rec_sc, thr_sc = precision_recall_curve(S_true_flat, S_prob_flat)
            f1_sc = 2*prec_sc*rec_sc/(prec_sc+rec_sc+1e-9)
            best_idx_sc = int(np.argmax(f1_sc))
            best_thr_scatter = thr_sc[best_idx_sc] if best_idx_sc < len(thr_sc) else 0.5
            auprc_scatter = average_precision_score(S_true_flat, S_prob_flat)
            try:
                auroc_scatter = roc_auc_score(S_true_flat, S_prob_flat)
            except:
                auroc_scatter = float("nan")
        else:
            best_thr_scatter = 0.5
            auprc_scatter, auroc_scatter = float("nan"), float("nan")
    
        # Pixel-level scatter metrics at chosen threshold
        scatter_met = scatter_metrics(all_S_logits, all_S_star, threshold=best_thr_scatter)
        pulse_scatter_met = scatter_metrics_per_pulse(all_S_logits, all_S_star, threshold=best_thr_scatter)
    
        # ======= Primary: PR-threshold with avg-preds/pulse constraint + Top-K fallback =======
        all_logitsK = torch.cat(all_logitsK, dim=0)  # (M,K)
        all_idxK    = torch.cat(all_idxK,    dim=0)  # (M,K)
        B, K = all_logitsK.shape
        _nan_report("all_primary_logitsK", all_logitsK)
    
        labels_primary, scores_primary = [], []
        TEMP = 0.9  # mild calibration
        for b in range(B):
            gt = all_primary_pixels[b]
            gt_set = set(gt.cpu().numpy().tolist()) if torch.is_tensor(gt) else set(gt)
            for k in range(K):
                pix = int(all_idxK[b, k].item())
                labels_primary.append(1 if pix in gt_set else 0)
                s = torch.sigmoid(all_logitsK[b, k] / TEMP)
                s = torch.nan_to_num(s, nan=0.0, posinf=1.0, neginf=0.0)
                scores_primary.append(float(s.item()))
        labels_primary = np.asarray(labels_primary, dtype=int)
        scores_primary = np.nan_to_num(np.asarray(scores_primary), nan=0.0, posinf=1.0, neginf=0.0)
    
        best_thr_primary = 0.5
        auprc_primary_candidates = float("nan")
        f1_thr = prec_thr = rec_thr = 0.0
        avg_preds_per_pulse_thr = 0.0
    
        # ICS gate arrays (built once, reused in every threshold sweep call)
        ics_probs_arr  = np.array(y_score_ics, dtype=np.float32)   # (B,)
        ics_gate_mask  = ics_probs_arr > ics_gate_thr               # (B,) bool
        ics_probs_rep  = np.repeat(ics_probs_arr, K)                # (B*K,) for flat ops

        # ── Gate diagnostics ──────────────────────────────────────────────
        y_true_ics_arr = np.array(y_true_ics, dtype=int)
        n_total_pulses = len(ics_gate_mask)
        n_pass         = ics_gate_mask.sum()
        n_block        = n_total_pulses - n_pass
        n_ics_pass     = (ics_gate_mask & (y_true_ics_arr == 1)).sum()
        n_ics_block    = ((~ics_gate_mask) & (y_true_ics_arr == 1)).sum()
        n_nics_pass    = (ics_gate_mask & (y_true_ics_arr == 0)).sum()
        n_nics_block   = ((~ics_gate_mask) & (y_true_ics_arr == 0)).sum()
        print(f"\nICS Gate diagnostics (thr={ics_gate_thr:.3f}):")
        print(f"  PASS : {n_pass:5d} pulses "
              f"({n_ics_pass} ICS ✓  +  {n_nics_pass} non-ICS ✗)")
        print(f"  BLOCK: {n_block:5d} pulses "
              f"({n_ics_block} ICS missed  +  {n_nics_block} non-ICS ✓)")
        if n_pass > 0:
            gate_precision = n_ics_pass / n_pass
            print(f"  Gate precision: {gate_precision:.3f} "
                  f"({n_ics_pass}/{n_pass} pulses passés sont ICS)")
        # ─────────────────────────────────────────────────────────────────

        if len(set(labels_primary)) > 1:
            prec_p, rec_p, thr_p = precision_recall_curve(labels_primary, scores_primary)
            f1_p = 2*prec_p*rec_p/(prec_p+rec_p+1e-9)

            K_pred_max = 2  # limit average #preds per pulse
            scores_matrix = scores_primary.reshape(B, K)  # (B, K) — vectorized

            def evaluate_threshold(th):
                # Gated logits already encode ICS probability — no external gate needed
                preds_flat = (scores_primary > th).astype(int)
                TP = ((labels_primary==1) & (preds_flat==1)).sum()
                FP = ((labels_primary==0) & (preds_flat==1)).sum()
                FN = ((labels_primary==1) & (preds_flat==0)).sum()
                prec = TP/(TP+FP+1e-9); rec = TP/(TP+FN+1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)
                avg_pp = float((scores_matrix > th).sum(axis=1).mean())
                return f1, prec, rec, avg_pp

            # subsample thresholds to avoid sweeping thousands
            thr_sample = thr_p[::max(1, len(thr_p)//200)]

            f1_best, thr_best, pr_best, rc_best, avg_pp_best = -1, 0.0, 0.0, 0.0, 0.0
            for t in thr_sample:
                f1x, px, rx, avg_pp = evaluate_threshold(t)
                if avg_pp <= K_pred_max and f1x > f1_best:
                    f1_best, thr_best, pr_best, rc_best, avg_pp_best = f1x, t, px, rx, avg_pp

            if f1_best < 0:  # no threshold meets constraint → take unconstrained best
                best_idx_p = int(np.argmax(f1_p))
                thr_best = thr_p[best_idx_p] if best_idx_p < len(thr_p) else 0.5
                f1x, px, rx, avg_pp = evaluate_threshold(thr_best)
                f1_best, pr_best, rc_best, avg_pp_best = f1x, px, rx, avg_pp
    
            best_thr_primary = float(thr_best)
            f1_thr, prec_thr, rec_thr = f1_best, pr_best, rc_best
            avg_preds_per_pulse_thr = avg_pp_best
            auprc_primary_candidates = average_precision_score(labels_primary, scores_primary)
    
        # --- Top-K fallback (pick the best over K=1..3) — vectorized ---
        pred_matrix = all_idxK.cpu().numpy()   # (B, K)
        gt_matrix   = np.full((B, K), -1, dtype=np.int64)
        for b in range(B):
            gt = all_primary_pixels[b]
            gt_arr = gt.cpu().numpy() if torch.is_tensor(gt) else np.array(gt, dtype=np.int64)
            if len(gt_arr) > 0:
                gt_matrix[b, :min(len(gt_arr), K)] = gt_arr[:K]

        # Use max gated logit per pulse as gate signal (much stronger than ICS head alone)
        max_gated_prob = torch.sigmoid(all_logitsK / TEMP).max(dim=1).values.numpy()  # (B,)

        def primary_metrics_topK(Ksel=1, gate_thr=0.4):
            tp = fp = fn = 0
            for b in range(B):
                gt_set = set(gt_matrix[b][gt_matrix[b] >= 0].tolist())
                # Skip prediction if model's max confidence is below gate threshold
                if max_gated_prob[b] < gate_thr:
                    fn += len(gt_set)
                    continue
                pred_set = set(pred_matrix[b, :Ksel].tolist())
                tp += len(gt_set & pred_set)
                fp += len(pred_set - gt_set)
                fn += len(gt_set - pred_set)
            prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9); f1 = 2*prec*rec/(prec+rec+1e-9)
            return f1, prec, rec

        f1_k_best, p_k_best, r_k_best, K_sel_best = -1, 0, 0, 1
        best_gate_thr = 0.4
        for Ksel in [1, 2, 3]:
            for gt in [0.3, 0.4, 0.5, 0.6]:
                f1k, pk, rk = primary_metrics_topK(Ksel, gate_thr=gt)
                if f1k > f1_k_best:
                    f1_k_best, p_k_best, r_k_best, K_sel_best = f1k, pk, rk, Ksel
                    best_gate_thr = gt
    
        use_topk = (f1_k_best > f1_thr)
        if use_topk:
            best_primary_strategy = f"Top-{K_sel_best} (gate@{best_gate_thr:.1f})"
            avg_recall, avg_precision, avg_f1 = r_k_best, p_k_best, f1_k_best
        else:
            best_primary_strategy = f"Thresh@{best_thr_primary:.3f} (avg preds/pulse≈{avg_preds_per_pulse_thr:.2f})"
            avg_recall, avg_precision, avg_f1 = rec_thr, prec_thr, f1_thr
    
        # Top-1 and within-radius (unchanged)
        acc1 = primary_top1_accuracy(all_idxK, all_logitsK, all_primary_pixels)
        accr = acc_within_radius(all_idxK, all_logitsK, all_primary_pixels, radius_px=1.0)
        acc_top1 = acc1 if not math.isnan(acc1) else float("nan")
        acc_r    = accr if not math.isnan(accr) else float("nan")
    
        # ======= PRINT =======
        print(f"\n{'='*70}")
        print("VALIDATION METRICS")
        print('='*70)
    
        print(f"\nICS Detection (pulse-level):")
        print(f"  AUPRC: {auprc_ics:.4f} | Best Thr: {best_thr_ics:.3f}")
        print(f"  Gate (target recall≥{TARGET_ICS_RECALL:.0%}): Thr={ics_gate_thr:.3f} | "
              f"Recall={ics_recall_at_gate:.4f} | Precision={ics_prec_at_gate:.4f}")
    
        print(f"\nScatter Detection (pixel-level):")
        print(f"  AUPRC: {auprc_scatter:.4f} | AUROC: {auroc_scatter:.4f} | Best Thr: {best_thr_scatter:.3f}")
        print(f"  Accuracy: {scatter_met['scatter_accuracy']:.4f}")
        print(f"  Recall: {scatter_met['scatter_recall']:.4f} | Precision: {scatter_met['scatter_precision']:.4f}")
        print(f"  F1: {scatter_met['scatter_f1']:.4f} | Specificity: {scatter_met['scatter_specificity']:.4f}")
        print(f"  Confusion: TP={scatter_met['TP']:.0f}, FP={scatter_met['FP']:.0f}, "
              f"FN={scatter_met['FN']:.0f}, TN={scatter_met['TN']:.0f}")
    
        print(f"\nScatter Detection (pulse-level):")
        print(f"  Accuracy: {pulse_scatter_met['pulse_scatter_accuracy']:.4f}")
        print(f"  Recall: {pulse_scatter_met['pulse_scatter_recall']:.4f}")
        print(f"  Precision: {pulse_scatter_met['pulse_scatter_precision']:.4f}")
    
        print(f"\nPrimary Localization:")
        print(f"  AUPRC(cands): {auprc_primary_candidates:.4f} | Strategy: {best_primary_strategy}")
        print(f"  AccTop1: {acc_top1:.4f} | AccRad1: {acc_r:.4f}")
        print(f"  Recall: {avg_recall:.4f} | Precision: {avg_precision:.4f} | F1: {avg_f1:.4f}")

        corr_mae = corr_mae_sum / max(corr_count, 1)
        print(f"\nCorrection:")
        print(f"  MAE (corrected energy): {corr_mae:.4f}")
        print('='*70)
    
        return {
            # ICS
            "AUPRC_ICS": auprc_ics,
            "BestThr_ICS": float(best_thr_ics),
            "ICS_Gate_Thr": float(ics_gate_thr),
            "ICS_Recall_at_Gate": float(ics_recall_at_gate),
            "ICS_Precision_at_Gate": float(ics_prec_at_gate),
    
            # Scatter (pixel)
            "AUPRC_Scatter": auprc_scatter,
            "AUROC_Scatter": auroc_scatter,
            "Scatter_Accuracy": scatter_met['scatter_accuracy'],
            "Scatter_Recall": scatter_met['scatter_recall'],
            "Scatter_Precision": scatter_met['scatter_precision'],
            "Scatter_F1": scatter_met['scatter_f1'],
            "Scatter_Specificity": scatter_met['scatter_specificity'],
    
            # Scatter (pulse)
            "Pulse_Scatter_Accuracy": pulse_scatter_met['pulse_scatter_accuracy'],
            "Pulse_Scatter_Recall": pulse_scatter_met['pulse_scatter_recall'],
            "Pulse_Scatter_Precision": pulse_scatter_met['pulse_scatter_precision'],
    
            # Primary (using chosen strategy)
            "Primary_AccTop1": acc_top1,
            "Primary_AccRad1": acc_r,
            "Primary_Recall": avg_recall,
            "Primary_Precision": avg_precision,
            "Primary_F1": avg_f1,
            "BestThr_Primary": float(best_thr_primary),
            "AUPRC_PrimaryCandidates": float(auprc_primary_candidates),

            # Correction
            "Correction_MAE": corr_mae,
        }
    
    
        
    
    def save_ckpt(self, name, epoch, metrics):
        path = os.path.join(self.ckpt_dir, name)
        torch.save({
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "cfg": self.cfg,
        }, path)
        # Sauvegarde weights-only
        torch.save(self.model.state_dict(), 
                  os.path.join(self.ckpt_dir, name.replace(".pt","_weights.pt")))
        print(f"Saved checkpoint: {path}")
        
    def fit(self, dl_tr, dl_va, epochs=10, early_patience=5):
        patience = 0
        for ep in range(1, epochs + 1):
            print(f"\nEpoch {ep}/{epochs}")
            tr_loss = self.train_one_epoch(dl_tr, epoch=ep)
    
            # Évalue
            metrics = self.validate(dl_va)
    
            # === Sélection du score de référence ===
            # Composite: Primary_F1 pénalisé si ICS recall trop bas
            # On veut ICS recall >= 0.90. En dessous, on réduit le score linéairement.
            primary_f1  = metrics.get("Primary_F1", float("nan"))
            ics_recall  = metrics.get("ICS_Recall_at_Gate", float("nan"))
            if not math.isnan(primary_f1) and not math.isnan(ics_recall):
                ics_penalty = min(1.0, ics_recall / 0.80)   # 1.0 si recall>=0.80, sinon <1
                composite   = primary_f1 * ics_penalty
                picked_key  = "Primary_F1 * ICS_recall_penalty"
                score       = composite
            else:
                priority = ["Primary_F1", "Primary_AccRad1", "AUPRC_Scatter", "AUPRC_ICS"]
                picked_key, score = None, 0.0
                for k in priority:
                    v = metrics.get(k, float("nan"))
                    if v is not None and not math.isnan(v):
                        picked_key, score = k, float(v)
                        break
    
            print(f"[fit] Using score={score:.4f} from metric='{picked_key or 'none'}'")
    
            # LR scheduler step (ReduceLROnPlateau en mode 'max')
            self.scheduler.step(score)
    
            # Sauvegarde "last"
            self.save_ckpt("last.pt", ep, metrics)
    
            # Sauvegarde "best" si amélioration
            if score > self.best_score:
                self.best_score = score
                patience = 0
                self.save_ckpt("best.pt", ep, metrics)
                print(f"✓ New best score: {score:.4f} (metric: {picked_key})")
            else:
                patience += 1
                print(f"No improvement. Patience {patience}/{early_patience}.")
                if patience >= early_patience:
                    print("Early stopping triggered.")
                    break
    
        print(f"\nTraining completed. Best score: {self.best_score:.4f}")

