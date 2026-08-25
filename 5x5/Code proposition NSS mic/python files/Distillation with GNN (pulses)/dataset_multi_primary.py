
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Augmentation pour grilles 5x5 (flip / rotation) – cohérent entrées + labels
# ---------------------------------------------------------------------------
def _apply_spatial_augment(grid, flip_h, flip_v, rot90):
    """Applique flip H, flip V, rotation 90° à une grille 5x5 (ou 3,5,5)."""
    out = np.asarray(grid, dtype=np.float32)
    if flip_h:
        out = np.flip(out, axis=-1).copy()
    if flip_v:
        out = np.flip(out, axis=-2).copy()
    if rot90 != 0:
        # axis -2,-1 = row, col
        out = np.rot90(out, k=rot90, axes=(-2, -1)).copy()
    return out


def _indices_from_grid(grid_5x5):
    """Retourne la liste des indices (0..24) où grid[r,c] > 0."""
    rr, cc = np.where(np.asarray(grid_5x5) > 0)
    return (rr * 5 + cc).astype(np.int64).tolist()


class Augment5x5Wrapper(Dataset):
    """
    Wrapper qui applique une augmentation spatiale aléatoire (flip H/V, rot 90°)
    à chaque échantillon. À utiliser uniquement sur le sous-ensemble d'entraînement.
    """
    def __init__(self, dataset, prob=0.5):
        self.dataset = dataset
        self.prob = float(prob)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        x = x.numpy()
        E, T, P_input = x[0], x[1], x[2]
        S_star = y["S_star"].numpy()
        P_star = y["P_star"].numpy()
        E_star = y["E_star"].numpy()
        T_star = y["T_star"].numpy()

        rng = np.random.default_rng()
        flip_h = rng.random() < self.prob
        flip_v = rng.random() < self.prob
        rot90 = rng.integers(0, 4)

        E = _apply_spatial_augment(E, flip_h, flip_v, rot90)
        T = _apply_spatial_augment(T, flip_h, flip_v, rot90)
        P_input = _apply_spatial_augment(P_input, flip_h, flip_v, rot90)
        S_star = _apply_spatial_augment(S_star, flip_h, flip_v, rot90)
        P_star = _apply_spatial_augment(P_star, flip_h, flip_v, rot90)
        E_star = E.copy()
        T_star = T.copy()

        scatter_pixels = _indices_from_grid(S_star)
        primary_pixels = _indices_from_grid(P_star)

        x_new = np.stack([E, T, P_input], axis=0).astype(np.float32)
        y_new = {
            "S_star": torch.from_numpy(S_star).float(),
            "P_star": torch.from_numpy(P_star).float(),
            "E_star": torch.from_numpy(E_star).float(),
            "T_star": torch.from_numpy(T_star).float(),
            "is_ics": y["is_ics"],
            "scatter_pixels": torch.tensor(scatter_pixels, dtype=torch.long),
            "primary_pixels": torch.tensor(primary_pixels, dtype=torch.long),
            "pulse_id": y["pulse_id"],
        }
        return torch.from_numpy(x_new), y_new


class Prepared5x5Dataset(Dataset):
    """
    Dataset pour grilles 5x5 avec support multi-canal, primaires, scatters,
    assignment map et énergie corrigée.
    """
    def __init__(self, agg_dict, time_shift_min=True, augment=False, augment_prob=0.5):
        """
        Args:
            agg_dict: dict returned by aggregate_to_5x5() with keys:
                E, T, N_hits, E_var, E_min, is_ics,
                primary_pixels, scatter_pixels, assign_map, E_corrected, pulse_ids
        """
        self.E = np.asarray(agg_dict["E"], dtype=np.float32)
        self.T = np.asarray(agg_dict["T"], dtype=np.float32)
        self.N_hits = np.asarray(agg_dict["N_hits"], dtype=np.float32)
        self.E_var = np.asarray(agg_dict["E_var"], dtype=np.float32)
        self.E_min = np.asarray(agg_dict["E_min"], dtype=np.float32)
        self.is_ics = np.asarray(agg_dict["is_ics"], dtype=bool)
        self.primary_pixels = agg_dict["primary_pixels"]
        self.scatter_pixels = agg_dict["scatter_pixels"]
        self.assign_map = np.asarray(agg_dict["assign_map"], dtype=np.int32)
        self.E_corrected = np.asarray(agg_dict["E_corrected"], dtype=np.float32)
        self.pulse_ids = np.asarray(agg_dict["pulse_ids"], dtype=int)

        assert self.E.shape[1:] == (5, 5)
        self.time_shift_min = time_shift_min
        self.augment = bool(augment)
        self.augment_prob = float(augment_prob)

    def __len__(self):
        return len(self.E)

    def __getitem__(self, i):
        E = self.E[i].copy()
        T = self.T[i].copy()
        N_h = self.N_hits[i].copy()
        Evar = self.E_var[i].copy()
        Emin = self.E_min[i].copy()
        Ecorr = self.E_corrected[i].copy()
        amap = self.assign_map[i].copy().astype(np.float32)

        P_input = (E > 0).astype(np.float32)

        S_star = np.zeros((5, 5), dtype=np.float32)
        for pix_idx in self.scatter_pixels[i]:
            r, c = pix_idx // 5, pix_idx % 5
            S_star[r, c] = 1.0

        P_star = np.zeros((5, 5), dtype=np.float32)
        for pix_idx in self.primary_pixels[i]:
            r, c = pix_idx // 5, pix_idx % 5
            P_star[r, c] = 1.0

        if self.time_shift_min and P_input.any():
            tmin = T[P_input > 0].min()
            T = T - tmin

        if self.augment and self.augment_prob > 0:
            rng = np.random.default_rng()
            flip_h = rng.random() < self.augment_prob
            flip_v = rng.random() < self.augment_prob
            rot90 = rng.integers(0, 4)

            E = _apply_spatial_augment(E, flip_h, flip_v, rot90)
            T = _apply_spatial_augment(T, flip_h, flip_v, rot90)
            N_h = _apply_spatial_augment(N_h, flip_h, flip_v, rot90)
            Evar = _apply_spatial_augment(Evar, flip_h, flip_v, rot90)
            Emin = _apply_spatial_augment(Emin, flip_h, flip_v, rot90)
            P_input = _apply_spatial_augment(P_input, flip_h, flip_v, rot90)
            S_star = _apply_spatial_augment(S_star, flip_h, flip_v, rot90)
            P_star = _apply_spatial_augment(P_star, flip_h, flip_v, rot90)
            Ecorr = _apply_spatial_augment(Ecorr, flip_h, flip_v, rot90)
            amap = _apply_spatial_augment(amap, flip_h, flip_v, rot90)

            scatter_pixels = _indices_from_grid(S_star)
            primary_pixels = _indices_from_grid(P_star)
        else:
            scatter_pixels = self.scatter_pixels[i]
            primary_pixels = self.primary_pixels[i]

        # 6 input channels: E, T, active_mask, N_hits, E_var, E_min
        x = np.stack([E, T, P_input, N_h, Evar, Emin], axis=0).astype(np.float32)
        y = {
            "S_star": torch.from_numpy(S_star).float(),
            "P_star": torch.from_numpy(P_star).float(),
            "E_star": torch.from_numpy(E).float(),
            "T_star": torch.from_numpy(T).float(),
            "is_ics": torch.tensor(bool(self.is_ics[i])),
            "scatter_pixels": torch.tensor(scatter_pixels, dtype=torch.long),
            "primary_pixels": torch.tensor(primary_pixels, dtype=torch.long),
            "assign_map": torch.from_numpy(amap).float(),
            "E_corrected": torch.from_numpy(Ecorr).float(),
            "pulse_id": torch.tensor(int(self.pulse_ids[i])),
        }
        return torch.from_numpy(x), y

class PerEventDataset(Dataset):
    """
    Dataset per-event: chaque sample est un cluster (pulse_id, eventID).
    Le primaire est toujours un pixel unique (ou vide si non-ICS).
    Interface identique à Prepared5x5Dataset — compatible avec le même Trainer.
    """
    def __init__(self, E_list, T_list, is_ics_list, primary_pixels_list, scatter_pixels_list,
                 meta_list=None, time_shift_min=True, augment=False, augment_prob=0.5):
        self.E = np.asarray(E_list, dtype=np.float32)
        self.T = np.asarray(T_list, dtype=np.float32)
        self.is_ics = np.asarray(is_ics_list, dtype=bool)
        self.primary_pixels = primary_pixels_list
        self.scatter_pixels = scatter_pixels_list
        self.meta = meta_list if meta_list is not None else [
            {"pulse_id": i, "event_id": 0} for i in range(len(E_list))
        ]
        assert self.E.shape[1:] == (5, 5)
        self.time_shift_min = time_shift_min
        self.augment = bool(augment)
        self.augment_prob = float(augment_prob)

    def __len__(self):
        return len(self.E)

    def __getitem__(self, i):
        E = self.E[i].copy()
        T = self.T[i].copy()
        P_input = (E > 0).astype(np.float32)

        S_star = np.zeros((5, 5), dtype=np.float32)
        for pix_idx in self.scatter_pixels[i]:
            r, c = pix_idx // 5, pix_idx % 5
            S_star[r, c] = 1.0

        P_star = np.zeros((5, 5), dtype=np.float32)
        for pix_idx in self.primary_pixels[i]:
            r, c = pix_idx // 5, pix_idx % 5
            P_star[r, c] = 1.0

        if self.time_shift_min and P_input.any():
            tmin = T[P_input > 0].min()
            T = T - tmin

        if self.augment and self.augment_prob > 0:
            rng = np.random.default_rng()
            flip_h = rng.random() < self.augment_prob
            flip_v = rng.random() < self.augment_prob
            rot90 = rng.integers(0, 4)
            E        = _apply_spatial_augment(E,       flip_h, flip_v, rot90)
            T        = _apply_spatial_augment(T,       flip_h, flip_v, rot90)
            P_input  = _apply_spatial_augment(P_input, flip_h, flip_v, rot90)
            S_star   = _apply_spatial_augment(S_star,  flip_h, flip_v, rot90)
            P_star   = _apply_spatial_augment(P_star,  flip_h, flip_v, rot90)
            scatter_pixels = _indices_from_grid(S_star)
            primary_pixels = _indices_from_grid(P_star)
        else:
            scatter_pixels = self.scatter_pixels[i]
            primary_pixels = self.primary_pixels[i]

        # --- Compute assign_map: for each scatter pixel, index of nearest primary ---
        assign_map = np.full((5, 5), -1, dtype=np.float32)
        if len(primary_pixels) > 0 and len(scatter_pixels) > 0:
            prim_coords = np.array([(p // 5, p % 5) for p in primary_pixels], dtype=np.float32)
            for sp in scatter_pixels:
                sr, sc = sp // 5, sp % 5
                dists = np.sqrt((prim_coords[:, 0] - sr)**2 + (prim_coords[:, 1] - sc)**2)
                assign_map[sr, sc] = int(np.argmin(dists))

        # --- E_corrected: energy with scatter pixels zeroed out ---
        E_corrected = E.copy()
        for sp in scatter_pixels:
            sr, sc = sp // 5, sp % 5
            E_corrected[sr, sc] = 0.0

        x = np.stack([E, T, P_input], axis=0).astype(np.float32)
        y = {
            "S_star":         torch.from_numpy(S_star).float(),
            "P_star":         torch.from_numpy(P_star).float(),
            "E_star":         torch.from_numpy(E).float(),
            "T_star":         torch.from_numpy(T).float(),
            "is_ics":         torch.tensor(bool(self.is_ics[i])),
            "scatter_pixels": torch.tensor(scatter_pixels, dtype=torch.long),
            "primary_pixels": torch.tensor(primary_pixels, dtype=torch.long),
            "assign_map":     torch.from_numpy(assign_map).float(),
            "E_corrected":    torch.from_numpy(E_corrected).float(),
            "pulse_id":       torch.tensor(int(self.meta[i]["pulse_id"])),
            "event_id":       torch.tensor(int(self.meta[i]["event_id"])),
        }
        return torch.from_numpy(x), y


class AggregatedFrame5x5Dataset(Dataset):
    """
    Dataset agrégé avec support de MULTIPLES pixels primaires par pulse.
    """
    def __init__(self, df, time_shift_min=True):
        self.df = df.copy()
        self.H = self.W = 5
        self.time_shift_min = time_shift_min
        self.df["pulse_id"] = self.df["pulse_id"].astype(int)
        self.pulse_ids = np.sort(self.df["pulse_id"].unique())
        
        # Déterminer is_ics par pulse
        if "is_ics_row" in self.df.columns:
            is_ics_by_pulse = self.df.groupby("pulse_id")["is_ics_row"].any()
        else:
            is_ics_by_pulse = self.df.groupby("pulse_id")["edep_sum"].apply(lambda s: bool((s > 0).any()))
        self.is_ics_by_pulse = is_ics_by_pulse.to_dict()
        
        # Déterminer primary_pixels par pulse (MULTIPLES possibles)
        self.primary_pixels_by_pulse = {}
        if {"primary_row", "primary_col"}.issubset(self.df.columns):
            # Si multiples lignes avec primary_row/col, on les prend toutes
            for pid, g in self.df.groupby("pulse_id"):
                primaries = set()
                for _, row in g.iterrows():
                    pr = int(row["primary_row"])
                    pc = int(row["primary_col"])
                    if pr >= 0 and pc >= 0:
                        primaries.add(pr * 5 + pc)
                self.primary_pixels_by_pulse[int(pid)] = list(primaries)
        else:
            # Fallback : utiliser is_ics_row si disponible
            if "is_ics_row" in self.df.columns:
                for pid, g in self.df.groupby("pulse_id"):
                    gics = g[g["is_ics_row"] == True]
                    primaries = []
                    if len(gics) > 0:
                        for idx in gics["time_min"].nsmallest(3).index:  # Top 3 plus précoces
                            r, c = int(self.df.loc[idx, "row"]), int(self.df.loc[idx, "col"])
                            primaries.append(r * 5 + c)
                    self.primary_pixels_by_pulse[int(pid)] = primaries
            else:
                for pid in self.pulse_ids:
                    self.primary_pixels_by_pulse[int(pid)] = []
    
    def __len__(self): 
        return len(self.pulse_ids)
    
    def __getitem__(self, ix):
        pid = int(self.pulse_ids[ix])
        g = self.df[self.df["pulse_id"] == pid]
        E = np.zeros((self.H, self.W), np.float32)
        T = np.zeros((self.H, self.W), np.float32)
        
        for _, r in g.iterrows():
            rr, cc = int(r["row"]), int(r["col"])
            E[rr, cc] = float(r["edep_sum"])
            T[rr, cc] = float(r["time_min"])
        
        # P_input : masque de tous les pixels avec énergie
        P_input = (E > 0).astype(np.float32)
        
        # P_star : grille multi-label avec 1 pour CHAQUE pixel primaire
        P_star = np.zeros((self.H, self.W), dtype=np.float32)
        primary_pix_list = self.primary_pixels_by_pulse.get(pid, [])
        for pix_idx in primary_pix_list:
            r_prim = pix_idx // 5
            c_prim = pix_idx % 5
            P_star[r_prim, c_prim] = 1.0
        
        # Time shift sur les pixels non-nuls
        if self.time_shift_min and P_input.any():
            tmin = T[P_input > 0].min()
            T = T - tmin
        
        # Input : [E, T, P_input] - 3 canaux
        x = np.stack([E, T, P_input], axis=0).astype(np.float32)
        
        y = {
            "P_star": torch.from_numpy(P_star).float(),      # Multi-label
            "E_star": torch.from_numpy(E).float(),
            "T_star": torch.from_numpy(T).float(),
            "is_ics": torch.tensor(bool(self.is_ics_by_pulse.get(pid, False))),
            "primary_pixels": torch.tensor(primary_pix_list, dtype=torch.long),  # Liste
            "pulse_id": torch.tensor(pid),
        }
        return torch.from_numpy(x), y
