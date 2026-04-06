"""
train.py — Full training pipeline (per-event approach)
"""

import os, random, sys, pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.amp import GradScaler

sys.path.insert(0, os.path.dirname(__file__))

from dataset_multi_primary import PerEventDataset
from model_V2 import ExampleTrunk, PhysFormerWrapper
from loss_multi_primary import build_dist_mm
from trainer_multi_primary import Trainer

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CFG = {
    "data": {
        "csv_paths": ["labeled_pulse_data.csv"],
        "pixel_pitch_mm": (2.0, 2.0),
    },
    "model": {
        "embed_dim": 96,
        "heads": 4,
        "mhsa_layers": 1,
        "topk": 8,
        "confinement_radius_px": 2.0,
        "primary_cap": 0.38,
        "primary_cap_tol": 0.02,
        "sigma_t_ns": 0.2,
        "c_mm_per_ns": 299.792458,
        "alpha_tof": 3.0,
        "alpha_spatial": 0.1,
    },
    "train": {
        "batch_size": 64,
        "num_workers": 0,
        "epochs": 30,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "warm_T_epochs": 3,
        "val_split": 0.2,
        "seed": 1337,
    },
    "mixed_precision": True,
    "ckpt_dir": "./checkpoints",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────
# Collate: handles variable-length pixel lists
# ─────────────────────────────────────────────
def collate_fn(batch):
    xs, ys = zip(*batch)
    x = torch.stack(xs)
    y = {}
    for k in ys[0]:
        vals = [s[k] for s in ys]
        if isinstance(vals[0], torch.Tensor) and vals[0].ndim == 0:
            y[k] = torch.stack(vals)
        elif isinstance(vals[0], torch.Tensor) and vals[0].shape == (5, 5):
            y[k] = torch.stack(vals)
        else:
            y[k] = vals  # variable-length: keep as list
    return x, y


# ─────────────────────────────────────────────
# Pos-weight estimation directly from arrays
# ─────────────────────────────────────────────
def estimate_pos_weight_from_lists(pixel_lists, train_indices, n_total_pixels, cap=200.0):
    """Compute neg/pos ratio directly from pixel index lists — no DataLoader needed."""
    pos = sum(len(pixel_lists[i]) for i in train_indices)
    neg = n_total_pixels - pos
    ratio = neg / max(pos, 1)
    return torch.tensor([min(ratio, cap)])


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    set_seed(CFG["train"]["seed"])
    print(f"Device: {DEVICE}")

    # ── 1. Load from cache ────────────────────
    script_dir  = os.path.dirname(__file__)
    cache_npz   = os.path.join(script_dir, "data_cache.npz")
    cache_lists = os.path.join(script_dir, "data_cache_lists.pkl")

    if not os.path.exists(cache_npz) or not os.path.exists(cache_lists):
        print("\nCache not found. Run preprocess.py first:")
        print("  python3 preprocess.py")
        sys.exit(1)

    print("\n[1/4] Loading data from cache...")
    import time; t0 = time.time()
    arrays      = np.load(cache_npz)
    E_list      = list(arrays["E"])       # list of (5,5) arrays
    T_list      = list(arrays["T"])
    isics_list  = arrays["isics"].tolist()

    with open(cache_lists, "rb") as f:
        lists       = pickle.load(f)
    prim_list    = lists["prim_list"]
    scatter_list = lists["scatter_list"]
    meta_list    = lists["meta_list"]

    n_total     = len(E_list)
    n_ics       = sum(isics_list)
    n_primaries = sum(len(p) > 0 for p in prim_list)
    print(f"  Loaded {n_total:,} events in {time.time()-t0:.1f}s")
    print(f"  ICS events   : {n_ics:,} ({100*n_ics/n_total:.1f}%)")
    print(f"  With primary : {n_primaries:,}")

    # ── 2. Dataset & splits ───────────────────
    print("\n[2/4] Building dataset & splits...")
    dataset = PerEventDataset(
        E_list, T_list, isics_list, prim_list, scatter_list, meta_list,
        time_shift_min=True, augment=True, augment_prob=0.5
    )

    labels = np.array(isics_list, dtype=int)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["train"]["val_split"],
        random_state=CFG["train"]["seed"]
    )
    idx_tr, idx_va = next(sss.split(np.zeros(n_total), labels))

    ds_tr = Subset(dataset, idx_tr)
    ds_va = Subset(dataset, idx_va)
    print(f"  Train: {len(ds_tr):,} | Val: {len(ds_va):,}")

    dl_tr = DataLoader(ds_tr, batch_size=CFG["train"]["batch_size"],
                       shuffle=True,  num_workers=CFG["train"]["num_workers"],
                       collate_fn=collate_fn, pin_memory=(DEVICE == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=CFG["train"]["batch_size"],
                       shuffle=False, num_workers=CFG["train"]["num_workers"],
                       collate_fn=collate_fn, pin_memory=(DEVICE == "cuda"))

    print("  Estimating class weights...")
    n_tr_pixels = len(idx_tr) * 25
    pw_P = estimate_pos_weight_from_lists(prim_list,    idx_tr, n_tr_pixels)
    pw_S = estimate_pos_weight_from_lists(scatter_list, idx_tr, n_tr_pixels)
    n_ics_tr  = sum(isics_list[i] for i in idx_tr)
    n_nics_tr = len(idx_tr) - n_ics_tr
    pw_ICS = torch.tensor([n_nics_tr / max(n_ics_tr, 1)])
    print(f"  pos_weight P={pw_P.item():.1f}  S={pw_S.item():.1f}  ICS={pw_ICS.item():.1f}")

    # ── 3. Model ─────────────────────────────
    print("\n[3/4] Building model...")
    trunk = ExampleTrunk(in_ch=3, out_ch=64)
    model = PhysFormerWrapper(
        trunk,
        d=CFG["model"]["embed_dim"],
        heads=CFG["model"]["heads"],
        mhsa_layers=CFG["model"]["mhsa_layers"],
        topk=CFG["model"]["topk"],
        primary_cap=CFG["model"]["primary_cap"],
        primary_cap_tol=CFG["model"]["primary_cap_tol"],
        c_mm_per_ns=CFG["model"]["c_mm_per_ns"],
        sigma_t_ns=CFG["model"]["sigma_t_ns"],
        alpha_tof=CFG["model"]["alpha_tof"],
        alpha_spatial=CFG["model"]["alpha_spatial"],
        confinement_radius_px=CFG["model"]["confinement_radius_px"],
        in_ch=3,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW([
        {"params": [p for n, p in model.named_parameters() if "graph" not in n],
         "lr": CFG["train"]["lr"]},
        {"params": [p for n, p in model.named_parameters() if "graph" in n],
         "lr": CFG["train"]["lr"] * 3},
    ], weight_decay=CFG["train"]["weight_decay"])

    use_amp = (DEVICE == "cuda") and CFG["mixed_precision"]
    scaler  = GradScaler("cuda", enabled=use_amp)

    dist_mm = build_dist_mm(
        pitch_xy=CFG["data"]["pixel_pitch_mm"], device=DEVICE
    )

    # ── 4. Train ─────────────────────────────
    print("\n[4/4] Training...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        cfg=CFG,
        dist_mm=dist_mm,
        pos_weight_P=pw_P,
        pos_weight_S=pw_S,
        pos_weight_ICS=pw_ICS,
        ckpt_dir=CFG["ckpt_dir"],
    )

    trainer.fit(dl_tr, dl_va, epochs=CFG["train"]["epochs"], early_patience=5)


if __name__ == "__main__":
    main()
