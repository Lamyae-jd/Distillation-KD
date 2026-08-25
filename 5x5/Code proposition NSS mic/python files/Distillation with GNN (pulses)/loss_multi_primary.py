import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



def bce_hit_loss(P_logits, P_star, pos_weight=None):
    """
    BCE pour détection des pixels actifs.
    P_star peut maintenant avoir PLUSIEURS 1s (multi-label).
    """
    B = P_logits.shape[0]
    return F.binary_cross_entropy_with_logits(
        P_logits.view(B,-1), P_star.view(B,-1),
        pos_weight=pos_weight, reduction='mean'
    )

def l1_energy_loss(E_hat, E_star, mask=None):
    """Loss L1 sur l'énergie, relative."""
    if mask is None: 
        mask = (E_star > 0).float()
    
    epsilon = 0.1
    relative_error = (E_hat - E_star).abs() / (E_star + epsilon)
    
    num = (mask * relative_error).sum()
    den = mask.sum().clamp_min(1.0)
    
    return num / den

def l1_time_loss(T_hat, T_star, mask=None):
    """Loss L1 sur le temps."""
    if mask is None: mask = (T_star!=0).float()
    num = (mask * (T_hat - T_star).abs()).sum()
    den = mask.sum().clamp_min(1.0)
    return num/den

def energy_conservation_loss(E_hat, E_star):
    """Pénalise la différence d'énergie totale."""
    return (E_hat.flatten(1).sum(1) - E_star.flatten(1).sum(1)).abs().mean()

def build_dist_mm(pitch_xy=(2.0,2.0), device="cpu"):
    """Construit la matrice de distances en mm."""
    yy, xx = torch.meshgrid(torch.arange(5, device=device), torch.arange(5, device=device), indexing='ij')
    idx = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
    dxy = idx.unsqueeze(1) - idx.unsqueeze(0)
    dx_mm = dxy[...,1]*pitch_xy[0]; dy_mm = dxy[...,0]*pitch_xy[1]
    return torch.sqrt(dx_mm**2 + dy_mm**2)

def tof_consistency_loss(T_hat, dist_mm, c_mm_per_ns=299.792458, sigma_t_ns=0.2, P_mask=None):
    """Cohérence TOF entre pixels."""
    B = T_hat.shape[0]; T = T_hat.view(B,-1,1)
    dt = T - T.transpose(1,2)
    target = dist_mm / c_mm_per_ns
    resid = (dt - target)**2 / (sigma_t_ns**2)
    if P_mask is not None:
        M = P_mask.view(B,-1,1) * P_mask.view(B,1,-1)
        resid = resid * M
        denom = M.sum((1,2)).clamp_min(1.0)
        return (resid.sum((1,2))/denom).mean()
    return resid.mean()

def fraction_cap_loss(E_hat, primary_pixels_list, cap=0.38, tol=0.02, is_ics=None):
    """Pénalise si l'énergie d'un pixel primaire dépasse cap+tol."""
    B = E_hat.shape[0]
    E = E_hat.view(B,-1)
    Ew = E.sum(1).clamp_min(1e-6)
    
    total_viol = 0.0
    count = 0
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        if len(primary_pix) == 0:
            continue
        
        for pix in primary_pix:
            if pix < 0 or pix >= 25:
                continue
            Ep = E[b, pix]
            frac = Ep / Ew[b]
            viol = (frac - (cap+tol)).clamp_min(0.0)
            
            if is_ics is not None and not is_ics[b]:
                continue
            
            total_viol += viol**2
            count += 1
    
    # Retourner un tensor PyTorch
    if count == 0:
        return torch.tensor(0.0, device=E_hat.device, dtype=E_hat.dtype)
    return torch.tensor(total_viol / count, device=E_hat.device, dtype=E_hat.dtype)

def spatial_confinement_loss(P_prob, primary_pixels_list, radius_px=2.0):
    """Pénalise les pixels actifs loin de TOUS les primaires."""
    B = P_prob.shape[0]
    P = P_prob.view(B,25)
    device = P_prob.device
    
    yy, xx = torch.meshgrid(torch.arange(5, device=device), torch.arange(5, device=device), indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
    
    total_loss = torch.tensor(0.0, device=device, dtype=P_prob.dtype)
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        if len(primary_pix) == 0:
            continue
        
        min_dists = torch.full((25,), float('inf'), device=device)
        
        for pi in primary_pix:
            if pi < 0 or pi >= 25:
                continue
            cy, cx = coords[pi]
            dists = torch.sqrt((coords[:,0]-cy)**2 + (coords[:,1]-cx)**2)
            min_dists = torch.minimum(min_dists, dists)
        
        mask_out = (min_dists > radius_px).float()
        total_loss += (P[b] * mask_out).sum()
    
    return total_loss / B

def ce_primary_multilabel_loss(logitsK, idxK, primary_pixels_list):
    """
    BCE multi-label pour prédire PLUSIEURS primaires.
    
    Args:
        logitsK: (B, K) - logits pour chaque candidat
        idxK: (B, K) - indices des K candidats dans [0..24]
        primary_pixels_list: liste de B tensors de tailles variables
    """
    B, K = logitsK.shape
    device = logitsK.device
    
    # Construire target multi-label (B, K)
    target = torch.zeros((B, K), dtype=torch.float32, device=device)
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        if len(primary_pix) == 0:
            continue
        
        for pix in primary_pix:
            pix_item = pix.item() if torch.is_tensor(pix) else pix
            if pix_item < 0 or pix_item >= 25:
                continue
            # Trouver si ce pixel primaire est dans les K candidats
            where = torch.nonzero(idxK[b] == pix_item, as_tuple=False)
            if where.numel() > 0:
                target[b, where[0, 0]] = 1.0
    
    # BCE multi-label
    loss = F.binary_cross_entropy_with_logits(logitsK, target, reduction='none')
    
    # Ne pénaliser que les pulses ICS (avec au moins un primaire)
    has_primary = torch.tensor([len(p) > 0 for p in primary_pixels_list], device=device)
    
    if has_primary.any():
        loss = loss[has_primary].mean()
    else:
        loss = torch.tensor(0.0, device=device)
    
    return loss

# def focal_primary_loss(logitsK, idxK, primary_pixels_list, alpha=0.25, gamma=2.0):
#     B, K = logitsK.shape
#     device = logitsK.device
    
#     target = torch.zeros((B, K), dtype=torch.float32, device=device)
    
#     for b in range(B):
#         primary_pix = primary_pixels_list[b]
#         if len(primary_pix) == 0:
#             continue
        
#         for pix in primary_pix:
#             pix_item = pix.item() if torch.is_tensor(pix) else pix
#             if pix_item < 0 or pix_item >= 25:
#                 continue
#             where = torch.nonzero(idxK[b] == pix_item, as_tuple=False)
#             if where.numel() > 0:
#                 target[b, where[0, 0]] = 1.0
    
#     # Focal Loss
#     bce = F.binary_cross_entropy_with_logits(logitsK, target, reduction='none')
#     p = torch.sigmoid(logitsK)
#     pt = target * p + (1 - target) * (1 - p)
#     focal_weight = (1 - pt).pow(gamma)
    
#     if alpha is not None:
#         alpha_t = target * alpha + (1 - target) * (1 - alpha)
#         focal_weight = alpha_t * focal_weight
    
#     loss = focal_weight * bce
    
#     # # NOUVEAU : Régularisation anti-collapse
#     # # Pénalise les logits trop négatifs pour éviter le collapse
#     # mean_logit = logitsK.mean()
#     # collapse_penalty = torch.clamp(-(mean_logit + 0.5), min=0.0)  # Pénalise si mean < -0.5
#     # MODIFIE: Regularisation anti-collapse PLUS FORTE
#     mean_logit = logitsK.mean()
#     std_logit = logitsK.std()
    
#     # Penaliser si mean < -0.5 (collapse vers negatif)
#     collapse_penalty = torch.clamp(-(mean_logit + 0.5), min=0.0)
    
#     # Penaliser si std trop faible (tous les logits identiques)
#     diversity_penalty = torch.clamp(0.5 - std_logit, min=0.0)
    
#     # CHANGE: 0.1 -> 1.0 pour collapse, ajoute 0.5 pour diversite
#     return loss.mean() + 1.0 * collapse_penalty + 0.5 * diversity_penalty
    
#     return loss.mean() + 0.1 * collapse_penalty

def focal_primary_loss(logitsK, idxK, primary_pixels_list, alpha=0.25, gamma=2.0):
    """
    Focal loss pour primaires avec FORTE régularisation anti-collapse.
    
    Args:
        logitsK: (B, K) - logits candidats
        idxK: (B, K) - indices pixels candidats
        primary_pixels_list: liste de B tensors (ground truth)
        alpha: pondération classe positive
        gamma: facteur focal (plus élevé = focus sur erreurs difficiles)
    """
    B, K = logitsK.shape
    device = logitsK.device
    
    # 1. Construire targets multi-label
    target = torch.zeros((B, K), dtype=torch.float32, device=device)
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        if len(primary_pix) == 0:
            continue
        
        for pix in primary_pix:
            pix_item = pix.item() if torch.is_tensor(pix) else pix
            if pix_item < 0 or pix_item >= 25:
                continue
            where = torch.nonzero(idxK[b] == pix_item, as_tuple=False)
            if where.numel() > 0:
                target[b, where[0, 0]] = 1.0
    
    # 2. Focal Loss classique
    bce = F.binary_cross_entropy_with_logits(logitsK, target, reduction='none')
    p = torch.sigmoid(logitsK)
    pt = target * p + (1 - target) * (1 - p)
    focal_weight = (1 - pt).pow(gamma)
    
    if alpha is not None:
        alpha_t = target * alpha + (1 - target) * (1 - alpha)
        focal_weight = alpha_t * focal_weight
    
    loss_focal = (focal_weight * bce).mean()
    
    #  3. RÉGULARISATIONS ANTI-COLLAPSE RENFORCÉES
    
    # 3a. Pénaliser logits trop négatifs (collapse)
    mean_logit = logitsK.mean()
    collapse_penalty = torch.clamp(-mean_logit - 0.5, min=0.0)  # Pénalise si mean < -0.5
    
    # 3b. Pénaliser variance trop faible (tous logits identiques)
    std_logit = logitsK.std()
    diversity_penalty = torch.clamp(0.8 - std_logit, min=0.0)  # Pénalise si std < 0.8
    
    # 3c. NOUVEAU : Encourager au moins 20% de logits > 0
    pos_ratio = (logitsK > 0).float().mean()
    positivity_penalty = torch.clamp(0.2 - pos_ratio, min=0.0)
    
    # 3d. NOUVEAU : Pénaliser max logit trop faible
    max_logit = logitsK.max()
    max_penalty = torch.clamp(0.5 - max_logit, min=0.0)  # Encourage max > 0.5
    
    #  4. POIDS DES RÉGULARISATIONS (AUGMENTÉS)
    # Avant: collapse=1.0, diversity=0.5
    # Maintenant: beaucoup plus fort
    regularization = (
        2.0 * collapse_penalty +      # x2 plus fort
        1.0 * diversity_penalty +     # x2 plus fort  
        1.5 * positivity_penalty +    # NOUVEAU
        1.0 * max_penalty             # NOUVEAU
    )
    
    return loss_focal + regularization


#Helper for multilabel ranking loss for primary, since the model is good at ranking but now in selecting

def build_primary_candidate_targets(idxK, primary_pixels_list):
    """
        Build multilabel targets over the K candidates for each pulse.

        Args:
            idxK: LongTensor of shape (B, K)
                  Candidate pixel ids selected for each pulse.
            primary_pixels_list: list of length B
                  Each element is a tensor/list of true primary pixel ids for that pulse.

        Returns:
            targets: FloatTensor of shape (B, K), with 1 for true primary candidates, 0 otherwise.
        """


def topk_primary_loss(P_logits, primary_pixels_list, K=8, use_focal=True,
                      focal_alpha=0.5, focal_gamma=2.0):
    """
    Loss "top-K par pulse": on ne regarde que les K pixels avec les plus forts logits
    par pulse, et on pousse les vrais primaires à être dans ce top-K avec une proba haute.
    
    Args:
        P_logits: (B, 1, 5, 5) ou (B, 25) - logits primaire
        primary_pixels_list: liste de B tensors (indices 0..24)
        K: nombre de candidats par pulse (ex. 8)
        use_focal: si True, utilise focal_primary_loss; sinon ce_primary_multilabel_loss
    """
    B = P_logits.shape[0]
    P_flat = P_logits.view(B, -1)
    K = min(K, P_flat.size(1))
    topk_vals, idxK = torch.topk(P_flat, K, dim=1)  # (B, K)
    logitsK = topk_vals
    if use_focal:
        return focal_primary_loss(
            logitsK, idxK, primary_pixels_list,
            alpha=focal_alpha, gamma=focal_gamma
        )
    return ce_primary_multilabel_loss(logitsK, idxK, primary_pixels_list)


def ics_score_from_P(P_prob):
    """Score ICS basé sur la probabilité maximale."""
    p = torch.nan_to_num(P_prob.flatten(1), nan=0.0, posinf=1.0, neginf=0.0).clamp(0,1)
    return p.max(dim=1).values

def primary_recall_precision_f1(idxK, logitsK, primary_pixels_list, threshold=0.3):
    """
    Calcule recall, precision et F1 pour la détection multi-label des primaires.
    
    Returns:
        (recall, precision, f1) as floats
    """
    B, K = logitsK.shape
    device = logitsK.device
    
    # Prédictions (pixels avec prob > threshold)
    probs = torch.sigmoid(logitsK)
    pred_mask = probs > threshold  # (B, K)
    
    tp, fp, fn = 0, 0, 0
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        
        # Convertir en set pour comparaison facile
        if torch.is_tensor(primary_pix):
            gt_set = set(primary_pix.cpu().numpy().tolist())
        else:
            gt_set = set(primary_pix)
        
        if len(gt_set) == 0:
            # Pas de primaire GT - toute prédiction est FP
            fp += pred_mask[b].sum().item()
            continue
        
        # Pixels prédits
        pred_indices = idxK[b][pred_mask[b]].cpu().numpy()
        pred_set = set(pred_indices.tolist())
        
        # Comptage
        tp += len(gt_set & pred_set)
        fp += len(pred_set - gt_set)
        fn += len(gt_set - pred_set)
    
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return recall, precision, f1


def primary_top1_accuracy(idxK, logitsK, primary_pixels_list):
    """
    Accuracy : est-ce que le top-1 prédit est dans la liste des primaires GT ?
    """
    B = logitsK.shape[0]
    pred_in_flat = idxK.gather(1, torch.argmax(logitsK, dim=1, keepdim=True)).squeeze(1)
    
    correct = 0
    total = 0
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        if torch.is_tensor(primary_pix):
            gt_set = set(primary_pix.cpu().numpy().tolist())
        else:
            gt_set = set(primary_pix)
        
        if len(gt_set) == 0:
            continue
        
        total += 1
        pred = pred_in_flat[b].item()
        if pred in gt_set:
            correct += 1
    
    return (correct / total) if total > 0 else float("nan")


def acc_within_radius(idxK, logitsK, primary_pixels_list, radius_px=1.0):
    """
    Accuracy : est-ce que le top-1 prédit est à distance <= radius_px d'un primaire GT ?
    """
    device = logitsK.device
    yy, xx = torch.meshgrid(torch.arange(5, device=device), torch.arange(5, device=device), indexing='ij')
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1).float()
    
    pred_flat = idxK.gather(1, torch.argmax(logitsK, dim=1, keepdim=True)).squeeze(1)
    B = pred_flat.shape[0]
    
    ok = 0
    den = 0
    
    for b in range(B):
        primary_pix = primary_pixels_list[b]
        if torch.is_tensor(primary_pix):
            gt_list = primary_pix.cpu().numpy().tolist()
        else:
            gt_list = list(primary_pix)
        
        if len(gt_list) == 0:
            continue
        
        den += 1
        yx_p = coords[pred_flat[b]]
        
        # Distance minimale vers n'importe quel primaire GT
        min_dist = float('inf')
        for pi in gt_list:
            if pi < 0 or pi >= 25:
                continue
            yx_t = coords[pi]
            dist = torch.sqrt(((yx_p - yx_t)**2).sum()).item()
            min_dist = min(min_dist, dist)
        
        if min_dist <= radius_px:
            ok += 1
    
    return (ok/den) if den > 0 else float("nan")

def scatter_metrics(S_logits, S_star, threshold=0.5):
    """
    Calcule les métriques pour la prédiction scatter vs non-scatter.
    
    Args:
        S_logits: (B, 1, 5, 5) - logits pour scatter
        S_star: (B, 1, 5, 5) - ground truth scatter (0 ou 1)
        threshold: seuil de décision
    
    Returns:
        dict avec accuracy, recall, precision, f1, specificity
    """
    B = S_logits.shape[0]
    
    # Aplatir
    S_pred_prob = torch.sigmoid(S_logits).view(B, -1)  # (B, 25)
    S_true = S_star.view(B, -1)  # (B, 25)
    
    # Prédictions binaires
    S_pred = (S_pred_prob > threshold).float()
    
    # Métriques
    TP = ((S_true == 1) & (S_pred == 1)).sum().float()
    TN = ((S_true == 0) & (S_pred == 0)).sum().float()
    FP = ((S_true == 0) & (S_pred == 1)).sum().float()
    FN = ((S_true == 1) & (S_pred == 0)).sum().float()
    
    # Calculs
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-9)
    recall = TP / (TP + FN + 1e-9)  # Sensitivity
    precision = TP / (TP + FP + 1e-9)
    f1 = 2 * TP / (2 * TP + FP + FN + 1e-9)
    specificity = TN / (TN + FP + 1e-9)  # True Negative Rate
    
    return {
        "scatter_accuracy": accuracy.item(),
        "scatter_recall": recall.item(),
        "scatter_precision": precision.item(),
        "scatter_f1": f1.item(),
        "scatter_specificity": specificity.item(),
        "TP": TP.item(),
        "TN": TN.item(),
        "FP": FP.item(),
        "FN": FN.item()
    }


def scatter_auprc_auroc(S_logits, S_star):
    """
    Calcule AUPRC et AUROC pour la détection des scatters.
    
    Args:
        S_logits: (B, 1, 5, 5)
        S_star: (B, 1, 5, 5)
    
    Returns:
        (auprc, auroc) as floats
    """
    from sklearn.metrics import average_precision_score, roc_auc_score
    
    S_pred_prob = torch.sigmoid(S_logits).detach().cpu().numpy().flatten()
    S_true = S_star.detach().cpu().numpy().flatten().astype(int)
    
    # Vérifier qu'il y a les deux classes
    if len(np.unique(S_true)) < 2:
        return float('nan'), float('nan')
    
    try:
        auprc = average_precision_score(S_true, S_pred_prob)
        auroc = roc_auc_score(S_true, S_pred_prob)
    except:
        auprc = float('nan')
        auroc = float('nan')
    
    return auprc, auroc


def scatter_confusion_matrix_pixelwise(S_logits, S_star, threshold=0.5):
    """
    Matrice de confusion agrégée sur tous les pixels.
    
    Returns:
        dict avec TP, TN, FP, FN totaux
    """
    S_pred_prob = torch.sigmoid(S_logits).flatten()
    S_true = S_star.flatten()
    S_pred = (S_pred_prob > threshold).float()
    
    TP = ((S_true == 1) & (S_pred == 1)).sum().item()
    TN = ((S_true == 0) & (S_pred == 0)).sum().item()
    FP = ((S_true == 0) & (S_pred == 1)).sum().item()
    FN = ((S_true == 1) & (S_pred == 0)).sum().item()
    
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}


def scatter_metrics_per_pulse(S_logits, S_star, threshold=0.5):
    """
    Calcule les métriques au niveau pulse (au moins un scatter détecté).
    
    Returns:
        dict avec pulse_accuracy, pulse_recall, pulse_precision
    """
    B = S_logits.shape[0]
    
    S_pred_prob = torch.sigmoid(S_logits).view(B, -1)
    S_true = S_star.view(B, -1)
    
    # Au niveau pulse : a-t-il au moins 1 scatter ?
    has_scatter_gt = (S_true.sum(dim=1) > 0).float()  # (B,)
    has_scatter_pred = (S_pred_prob.max(dim=1).values > threshold).float()  # (B,)
    
    TP_pulse = ((has_scatter_gt == 1) & (has_scatter_pred == 1)).sum().float()
    TN_pulse = ((has_scatter_gt == 0) & (has_scatter_pred == 0)).sum().float()
    FP_pulse = ((has_scatter_gt == 0) & (has_scatter_pred == 1)).sum().float()
    FN_pulse = ((has_scatter_gt == 1) & (has_scatter_pred == 0)).sum().float()
    
    accuracy = (TP_pulse + TN_pulse) / B
    recall = TP_pulse / (TP_pulse + FN_pulse + 1e-9)
    precision = TP_pulse / (TP_pulse + FP_pulse + 1e-9)
    
    return {
        "pulse_scatter_accuracy": accuracy.item(),
        "pulse_scatter_recall": recall.item(),
        "pulse_scatter_precision": precision.item(),
    }


class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p   = torch.sigmoid(logits)
        pt  = targets * p + (1 - targets) * (1 - p)
        w   = (1 - pt).pow(self.gamma)
        if self.alpha is not None:
            alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            w = alpha_t * w
        loss = w * bce
        if self.reduction == "mean": return loss.mean()
        if self.reduction == "sum":  return loss.sum()
        return loss
def multilabel_ranking_loss(logitsK, idxK, primary_pixels_list, is_ics,
                            margin=2.0, neg_push=3.0):
    """
    Ranking loss for the GraphPrimaryHead.

    - ICS pulses  : margin ranking loss — push every (pos, neg) pair so that
                    logit_pos > logit_neg + margin.
    - non-ICS     : push ALL candidate logits below 0 (teach the model to abstain).

    This directly addresses the precision problem: the old focal loss with
    anti-collapse regularisation forced logits to stay positive even on
    non-ICS pulses, generating massive FP.

    Args:
        logitsK  : (B, K) raw logits from GraphPrimaryHead
        idxK     : (B, K) pixel indices of the K candidates
        primary_pixels_list : list of B tensors with true primary pixel ids
        is_ics   : (B,) bool or float tensor — 1 if the pulse has a primary
        margin   : ranking margin for ICS pairs (default 1.0)
        neg_push : non-ICS pulses are pushed to logit < -neg_push (default 0.5)
    """
    B, K = logitsK.shape
    device = logitsK.device
    is_ics_bool = is_ics.bool() if torch.is_tensor(is_ics) else torch.tensor(is_ics, dtype=torch.bool, device=device)

    loss = torch.tensor(0.0, device=device, dtype=logitsK.dtype)
    n = 0

    for b in range(B):
        if is_ics_bool[b]:
            # Build pos mask over the K candidates
            pos_mask = torch.zeros(K, dtype=torch.bool, device=device)
            for pix in primary_pixels_list[b]:
                pix_item = pix.item() if torch.is_tensor(pix) else int(pix)
                if pix_item < 0 or pix_item >= 25:
                    continue
                where = (idxK[b] == pix_item).nonzero(as_tuple=False)
                if where.numel() > 0:
                    pos_mask[where[0, 0]] = True

            neg_mask = ~pos_mask
            pos_logits = logitsK[b][pos_mask]   # (n_pos,)
            neg_logits = logitsK[b][neg_mask]   # (n_neg,)

            if pos_logits.numel() > 0 and neg_logits.numel() > 0:
                # all (pos, neg) pairs: push pos above neg by margin
                diff = pos_logits.unsqueeze(1) - neg_logits.unsqueeze(0)  # (n_pos, n_neg)
                loss = loss + F.relu(margin - diff).mean()
                n += 1
            elif pos_logits.numel() > 0:
                # all K candidates are positives — just push them up
                loss = loss + F.relu(-pos_logits).mean()
                n += 1
        else:
            # non-ICS: push ALL candidates below 0
            loss = loss + F.relu(logitsK[b] + neg_push).mean()
            n += 1

    return loss / max(n, 1)


def pairwise_rank_loss(P_logits, P_star, k_neg=5):
    """
    P_logits, P_star: (B,1,5,5)
    For each positive pixel, sample k negatives from the same sample and apply
    logistic pairwise loss: softplus(-(z_pos - z_neg)).
    Returns scalar loss (mean over pairs in the batch). Safe if no positives.
    """
    B, _, H, W = P_logits.shape
    device = P_logits.device
    loss_terms = []
    for b in range(B):
        z = P_logits[b, 0].view(-1)      # (25,)
        y = P_star[b, 0].view(-1) > 0.5  # bool
        pos_idx = torch.nonzero(y, as_tuple=False).flatten()
        neg_idx = torch.nonzero(~y, as_tuple=False).flatten()
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        # sample negatives for each positive
        k = min(k_neg, neg_idx.numel())
        neg_sel = neg_idx[torch.randint(0, neg_idx.numel(), (k,), device=device)]
        for p in pos_idx:
            z_pos = z[p]
            z_neg = z[neg_sel]  # (k,)
            loss_terms.append(F.softplus(-(z_pos - z_neg)).mean())
    if len(loss_terms) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(loss_terms).mean()


# ──────────────────────────────────────────────────────────────────────────────
# Assignment loss: scatter pixel → primary linking
# ──────────────────────────────────────────────────────────────────────────────

def assignment_loss(assign_targets, assign_map_gt, primary_pixels_list, S_star):
    """
    L2 loss between predicted primary coordinates and GT primary coordinates,
    masked to scatter pixels only.

    Args:
        assign_targets: (B, 2, 5, 5) predicted primary (y, x) for each pixel
        assign_map_gt:  (B, 5, 5) int — index into primary_pixels_list[b]
                        -1 = not scatter, 0..K-1 = assigned to k-th primary
        primary_pixels_list: list of B lists of primary pixel indices (0..24)
        S_star: (B, 1, 5, 5) scatter ground truth mask
    Returns:
        scalar loss
    """
    B = assign_targets.shape[0]
    device = assign_targets.device

    total_loss = torch.tensor(0.0, device=device, dtype=assign_targets.dtype)
    count = 0

    for b in range(B):
        primaries = primary_pixels_list[b]
        if len(primaries) == 0:
            continue

        # Build GT coordinate targets for each scatter pixel
        amap = assign_map_gt[b]  # (5, 5) int, -1 or 0..K-1
        s_mask = S_star[b, 0] if S_star.dim() == 4 else S_star[b]  # (5, 5)

        for r in range(5):
            for c in range(5):
                k = int(amap[r, c].item())
                if k < 0 or s_mask[r, c] < 0.5:
                    continue
                if k >= len(primaries):
                    continue
                prim_idx = primaries[k]
                prim_idx = prim_idx.item() if torch.is_tensor(prim_idx) else int(prim_idx)
                gt_y = prim_idx // 5
                gt_x = prim_idx % 5

                pred_y = assign_targets[b, 0, r, c]
                pred_x = assign_targets[b, 1, r, c]

                total_loss = total_loss + (pred_y - gt_y)**2 + (pred_x - gt_x)**2
                count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, dtype=assign_targets.dtype)
    return total_loss / count


# ──────────────────────────────────────────────────────────────────────────────
# Correction loss: predict scatter-free energy map
# ──────────────────────────────────────────────────────────────────────────────

def correction_loss(E_corrected_pred, E_corrected_gt, E_input):
    """
    L1 loss on corrected energy + energy conservation soft penalty.

    Args:
        E_corrected_pred: (B, 1, 5, 5) predicted corrected energy
        E_corrected_gt:   (B, 5, 5) ground truth corrected energy
        E_input:          (B, 1, 5, 5) original energy (for conservation)
    Returns:
        scalar loss
    """
    gt = E_corrected_gt.unsqueeze(1) if E_corrected_gt.dim() == 3 else E_corrected_gt
    mask = (gt > 0).float()  # only penalise where there's energy to correct

    # L1 on corrected pixels
    l1 = ((E_corrected_pred - gt).abs() * mask).sum() / mask.sum().clamp_min(1.0)

    # Soft energy conservation: corrected energy should not exceed input energy per pixel
    excess = F.relu(E_corrected_pred - E_input)  # penalise if corrected > input
    conservation = excess.mean()

    return l1 + 0.5 * conservation
