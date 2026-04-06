"""
preprocess.py — Run once to build the per-event dataset and save to disk.
Usage: python3 preprocess.py

Output: data_cache.npz  +  data_cache_lists.pkl
"""

import os, sys, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing_multi_primary import load_csvs, map_rows_cols_with_pixelID, aggregate_by_event

DATA_DIR    = "/data/jdil1901/Documents/ics/code/5x5"
CSV_PATHS   = [
    f"{DATA_DIR}/output_file_pulse2.csv",       # ~150K rows (original)
    f"{DATA_DIR}/output_file_pulse.csv",         # ~150K rows
    f"{DATA_DIR}/output_file_pulse1.csv",        # ~150K rows
    f"{DATA_DIR}/hits_corrected.csv",            # ~150K rows
    f"{DATA_DIR}/output_file_pulse_test.csv",    # ~1.69M rows (is_scatter derived)
    f"{DATA_DIR}/labeled_pulse_data.csv",        # ~702K rows
]
CACHE_NPZ   = "data_cache.npz"
CACHE_LISTS = "data_cache_lists.pkl"


def main():
    print("=== Preprocessing pipeline ===\n")

    # ── 1. Load CSV ───────────────────────────────────────────────────────
    print("[1/3] Loading CSV...")
    t0 = time.time()
    csv_dir   = os.path.dirname(__file__)
    csv_paths = CSV_PATHS
    df        = load_csvs(csv_paths)
    df_rc     = map_rows_cols_with_pixelID(df)
    print(f"      {len(df_rc):,} hits loaded in {time.time()-t0:.1f}s")

    # ── 2. Aggregate by event ─────────────────────────────────────────────
    print("[2/3] Aggregating by event...")
    t0 = time.time()
    E_list, T_list, isics_list, prim_list, scatter_list, meta_list = aggregate_by_event(df_rc)
    n = len(E_list)
    print(f"      {n:,} events in {time.time()-t0:.1f}s")
    print(f"      ICS events   : {sum(isics_list):,} ({100*sum(isics_list)/n:.1f}%)")
    print(f"      With primary : {sum(len(p)>0 for p in prim_list):,}")

    # ── 3. Save to disk ───────────────────────────────────────────────────
    print("[3/3] Saving cache to disk...")
    t0 = time.time()

    # Fixed-size arrays → .npz (fast numpy format)
    np.savez_compressed(
        os.path.join(csv_dir, CACHE_NPZ),
        E      = np.stack(E_list).astype(np.float32),   # (N, 5, 5)
        T      = np.stack(T_list).astype(np.float32),   # (N, 5, 5)
        isics  = np.array(isics_list, dtype=bool),       # (N,)
    )

    # Variable-length lists → .pkl
    with open(os.path.join(csv_dir, CACHE_LISTS), "wb") as f:
        pickle.dump({"prim_list": prim_list,
                     "scatter_list": scatter_list,
                     "meta_list": meta_list}, f)

    size_npz   = os.path.getsize(os.path.join(csv_dir, CACHE_NPZ))   / 1e6
    size_lists = os.path.getsize(os.path.join(csv_dir, CACHE_LISTS)) / 1e6
    print(f"      Saved {CACHE_NPZ} ({size_npz:.1f} MB)")
    print(f"      Saved {CACHE_LISTS} ({size_lists:.1f} MB)")
    print(f"      Done in {time.time()-t0:.1f}s")
    print("\nCache ready. Run train.py to start training.")


if __name__ == "__main__":
    main()
