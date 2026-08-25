"""
preprocess.py — Run once to build the per-PULSE dataset and save to disk.
Usage: python3 preprocess.py

Output: data_cache.npz  +  data_cache_lists.pkl

Mirrors the event-based preprocess.py from "Distillation with GNN/" but
aggregates by pulse_id (multi-primary per pulse) instead of (pulse_id, eventID).
"""

import os, sys, pickle, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from preprocessing_multi_primary import load_csvs, map_rows_cols_with_pixelID, aggregate_to_5x5

DATA_DIR    = "/data/jdil1901/Documents/ics/code/5x5"
CSV_PATHS   = [
    f"{DATA_DIR}/output_file_pulse2.csv",
    f"{DATA_DIR}/output_file_pulse.csv",
    f"{DATA_DIR}/output_file_pulse1.csv",
    f"{DATA_DIR}/hits_corrected.csv",
    f"{DATA_DIR}/output_file_pulse_test.csv",
    f"{DATA_DIR}/labeled_pulse_data.csv",
]
CACHE_NPZ   = "data_cache.npz"
CACHE_LISTS = "data_cache_lists.pkl"


def main():
    print("=== Pulse-based preprocessing pipeline ===\n")

    print("[1/3] Loading CSV...")
    t0 = time.time()
    df    = load_csvs(CSV_PATHS)
    df_rc = map_rows_cols_with_pixelID(df)
    print(f"      {len(df_rc):,} hits loaded in {time.time()-t0:.1f}s")

    print("[2/3] Aggregating by pulse (multi-primary v8)...")
    t0 = time.time()
    agg = aggregate_to_5x5(df_rc, set_primary_only_if_ics=True, max_debug_prints=5)

    E_arr        = agg["E"]                 # (N, 5, 5)
    T_arr        = agg["T"]                 # (N, 5, 5)
    isics_arr    = agg["is_ics"]            # (N,) bool
    prim_list    = agg["primary_pixels"]    # list[list[int]]
    scatter_list = agg["scatter_pixels"]    # list[list[int]]
    pulse_ids    = agg["pulse_ids"]         # (N,) int

    n = E_arr.shape[0]
    n_ics = int(isics_arr.sum())
    n_prim = sum(1 for p in prim_list if len(p) > 0)
    print(f"      {n:,} pulses in {time.time()-t0:.1f}s")
    print(f"      ICS pulses   : {n_ics:,} ({100*n_ics/max(n,1):.1f}%)")
    print(f"      With primary : {n_prim:,}")
    print(f"      Avg primaries per ICS pulse: "
          f"{np.mean([len(p) for p,i in zip(prim_list,isics_arr) if i]):.2f}")

    print("[3/3] Saving cache to disk...")
    t0 = time.time()
    csv_dir = os.path.dirname(__file__)

    np.savez_compressed(
        os.path.join(csv_dir, CACHE_NPZ),
        E     = E_arr.astype(np.float32),
        T     = T_arr.astype(np.float32),
        isics = isics_arr.astype(bool),
    )

    meta_list = [{"pulse_id": int(pid), "event_id": 0} for pid in pulse_ids]

    with open(os.path.join(csv_dir, CACHE_LISTS), "wb") as f:
        pickle.dump({"prim_list":    prim_list,
                     "scatter_list": scatter_list,
                     "meta_list":    meta_list}, f)

    size_npz   = os.path.getsize(os.path.join(csv_dir, CACHE_NPZ))   / 1e6
    size_lists = os.path.getsize(os.path.join(csv_dir, CACHE_LISTS)) / 1e6
    print(f"      Saved {CACHE_NPZ} ({size_npz:.1f} MB)")
    print(f"      Saved {CACHE_LISTS} ({size_lists:.1f} MB)")
    print(f"      Done in {time.time()-t0:.1f}s")
    print("\nCache ready. Run train.py to start training.")


if __name__ == "__main__":
    main()
