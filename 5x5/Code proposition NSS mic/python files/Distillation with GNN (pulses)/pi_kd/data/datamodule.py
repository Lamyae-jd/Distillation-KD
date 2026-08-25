"""
Data loading for PI-KD training.
Reuses the existing PerEventDataset and cache files from the teacher pipeline.
"""

import os
import sys
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from dataset_multi_primary import PerEventDataset


def collate_fn(batch):
    """Collate with variable-length pixel lists."""
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
            y[k] = vals
    return x, y


def load_data(cfg, verbose=True):
    """
    Load cached data and create train/val DataLoaders.

    Args:
        cfg: config dict with data.cache_npz, data.cache_lists, train.*

    Returns:
        dl_train, dl_val, dataset
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pikd_dir = os.path.dirname(script_dir)  # pi-kd/

    cache_npz = os.path.join(pikd_dir, cfg["data"]["cache_npz"])
    cache_lists = os.path.join(pikd_dir, cfg["data"]["cache_lists"])

    if not os.path.exists(cache_npz):
        raise FileNotFoundError(
            f"Cache not found: {cache_npz}\n"
            f"Run preprocess.py first to generate the cache."
        )

    if verbose:
        print(f"Loading data from {cache_npz} ...")

    arrays = np.load(cache_npz)
    E_list = list(arrays["E"])
    T_list = list(arrays["T"])
    isics_list = arrays["isics"].tolist()

    with open(cache_lists, "rb") as f:
        lists = pickle.load(f)
    prim_list = lists["prim_list"]
    scatter_list = lists["scatter_list"]
    meta_list = lists["meta_list"]

    n_total = len(E_list)
    n_ics = sum(isics_list)

    if verbose:
        print(f"  {n_total:,} events, {n_ics:,} ICS ({100*n_ics/n_total:.1f}%)")

    # Build dataset (no augmentation — we'll handle it separately if needed)
    dataset = PerEventDataset(
        E_list, T_list, isics_list, prim_list, scatter_list, meta_list,
        time_shift_min=True, augment=False,
    )

    # Train augmented dataset
    dataset_train = PerEventDataset(
        E_list, T_list, isics_list, prim_list, scatter_list, meta_list,
        time_shift_min=True, augment=True, augment_prob=0.5,
    )

    # Stratified split
    train_cfg = cfg["train"]
    labels = np.array(isics_list, dtype=int)
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=train_cfg["val_split"],
        random_state=train_cfg["seed"],
    )
    idx_tr, idx_va = next(sss.split(np.zeros(n_total), labels))

    ds_tr = Subset(dataset_train, idx_tr)
    ds_va = Subset(dataset, idx_va)  # no augmentation for validation

    dl_tr = DataLoader(
        ds_tr,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 0),
        collate_fn=collate_fn,
        pin_memory=True,
    )

    if verbose:
        print(f"  Train: {len(ds_tr):,} | Val: {len(ds_va):,}")

    return dl_tr, dl_va, dataset
