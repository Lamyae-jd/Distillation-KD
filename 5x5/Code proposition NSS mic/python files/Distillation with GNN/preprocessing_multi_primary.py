

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List

PITCH_X_MM = 2.0
PITCH_Y_MM = 2.0

def load_csvs(csv_paths) -> pd.DataFrame:
    """Charge un ou plusieurs CSVs et retourne un DataFrame unifié.

    - Dérive is_scatter depuis nCrystalCompton/nCrystalRayleigh si absent.
    - Offset les pulse_id par fichier pour éviter les collisions entre CSVs.
    """
    cols = ["pulse_id","pixelID","posX","posY","edep","time_ns",
            "processName","nCrystalCompton","nCrystalRayleigh","parentID","trackID",
            "eventID","is_scatter"]
    frames = []
    pulse_id_offset = 0
    for p in csv_paths:
        print(f"  Loading {p} ...")
        df = pd.read_csv(p)
        keep = [c for c in cols if c in df.columns]
        if not keep:
            print(f"    -> skipped (no matching columns)")
            continue
        df = df[keep].copy()

        # Derive is_scatter if missing but Compton/Rayleigh columns exist
        if "is_scatter" not in df.columns:
            if "nCrystalCompton" in df.columns and "nCrystalRayleigh" in df.columns:
                df["nCrystalCompton"] = pd.to_numeric(df["nCrystalCompton"], errors="coerce").fillna(0)
                df["nCrystalRayleigh"] = pd.to_numeric(df["nCrystalRayleigh"], errors="coerce").fillna(0)
                df["is_scatter"] = ((df["nCrystalCompton"] + df["nCrystalRayleigh"]) > 0).astype(int)
                print(f"    -> derived is_scatter from nCrystalCompton + nCrystalRayleigh")
            else:
                print(f"    -> WARNING: no is_scatter and no Compton/Rayleigh columns, skipping {p}")
                continue

        # Skip files without pulse_id (raw GATE format, not pulse-grouped)
        if "pulse_id" not in df.columns:
            print(f"    -> skipped (no pulse_id column)")
            continue

        # Offset pulse_id to avoid collisions between files
        df["pulse_id"] = pd.to_numeric(df["pulse_id"], errors="coerce")
        df = df.dropna(subset=["pulse_id"])
        df["pulse_id"] = df["pulse_id"].astype(int) + pulse_id_offset
        max_pid = df["pulse_id"].max()
        pulse_id_offset = int(max_pid) + 1 if len(df) > 0 else pulse_id_offset

        print(f"    -> {len(df)} rows, pulse_id range [{df['pulse_id'].min()}, {df['pulse_id'].max()}]")
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=cols)

    df_all = pd.concat(frames, ignore_index=True)

    # types numériques
    for c in ["pulse_id","pixelID","nCrystalCompton","nCrystalRayleigh","parentID","trackID"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")
    for c in ["posX","posY","edep","time_ns"]:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    df_all = df_all.dropna(subset=["pulse_id","edep","time_ns"])
    df_all = df_all[df_all["edep"] > 0]
    df_all["pulse_id"] = df_all["pulse_id"].astype(int)

    print(f"  Total: {len(df_all)} rows, {df_all['pulse_id'].nunique()} unique pulses")
    return df_all.reset_index(drop=True)


def map_rows_cols_with_pixelID(df: pd.DataFrame,
                               pixelid_to_rc: Dict[int, Tuple[int,int]] = None) -> pd.DataFrame:
    """Ajoute colonnes row,col en utilisant pixelID (alignement d'index sécurisé)."""
    if "pixelID" not in df.columns:
        raise ValueError("pixelID column missing; use posX/posY mapping instead.")

    df = df.copy()

    if pixelid_to_rc is None:
        # hypothèse par défaut: row-major 0..24 (5x5)
        def to_rc(pid: int) -> Tuple[int,int]:
            pid = int(pid)
            return pid // 5, pid % 5
    else:
        def to_rc(pid: int) -> Tuple[int,int]:
            pid = int(pid)
            if pid not in pixelid_to_rc:
                raise KeyError(f"pixelID {pid} not in pixelid_to_rc map")
            return pixelid_to_rc[pid]

    pix = df["pixelID"].dropna().astype(int)
    rcs = pix.map(to_rc)  # conserve l'index

    row_series = pd.Series(index=df.index, dtype="Int64")
    col_series = pd.Series(index=df.index, dtype="Int64")
    row_series.loc[rcs.index] = [rc[0] for rc in rcs]
    col_series.loc[rcs.index] = [rc[1] for rc in rcs]

    df["row"] = row_series.clip(0, 4)
    df["col"] = col_series.clip(0, 4)
    return df


def map_rows_cols_with_positions(df: pd.DataFrame,
                                 pitch_x_mm=2.0, pitch_y_mm=2.0) -> pd.DataFrame:
    """Reconstruit row,col à partir de posX,posY (origine par pulse)."""
    if not {"posX","posY","pulse_id"}.issubset(df.columns):
        raise ValueError("posX/posY/pulse_id manquent; utilisez pixelID.")

    def rc_from_xy(g):
        x0 = (g["posX"].min() // pitch_x_mm) * pitch_x_mm
        y0 = (g["posY"].min() // pitch_y_mm) * pitch_y_mm
        col = np.rint((g["posX"] - x0) / pitch_x_mm).astype(int)
        row = np.rint((g["posY"] - y0) / pitch_y_mm).astype(int)
        col = np.clip(col, 0, 4)
        row = np.clip(row, 0, 4)
        g = g.copy()
        g["row"] = row
        g["col"] = col
        return g

    return df.groupby("pulse_id", group_keys=False).apply(rc_from_xy)


def identify_primary_pixels_v2(g: pd.DataFrame, 
                               energy_frac_range=(0.28, 0.48),
                               spatial_radius_px=2.5) -> List[int]:
    """
    Identifie les pixels primaires dans un pulse ICS (règle parentID).
    """
    has_scatter = ((g.get("nCrystalCompton", 0) > 0).any() or 
                   (g.get("nCrystalRayleigh", 0) > 0).any())
    if not has_scatter:
        return []

    if "parentID" not in g.columns:
        print("Warning: parentID column missing, falling back to heuristic")
        return identify_primary_pixels_fallback(g, energy_frac_range)

    g_candidates = g[g["parentID"] == 0].copy()
    if "processName" in g.columns:
        g_candidates = g_candidates[g_candidates["processName"] != "PhotoElectric"]
    if len(g_candidates) == 0:
        return []

    total_E = g["edep"].sum()

    g_scatter = g[(g.get("nCrystalCompton", 0) > 0) | (g.get("nCrystalRayleigh", 0) > 0)]
    scatter_positions = g_scatter[["row", "col"]].values if len(g_scatter) > 0 else np.empty((0,2))

    primary_pixels = []
    for (r, c), grp_pix in g_candidates.groupby(["row", "col"]):
        pixel_E = grp_pix["edep"].sum()
        frac = pixel_E / total_E
        if not (energy_frac_range[0] <= frac <= energy_frac_range[1]):
            continue

        if scatter_positions.size:
            distances = np.sqrt((scatter_positions[:, 0] - r)**2 + (scatter_positions[:, 1] - c)**2)
            if distances.min() > spatial_radius_px:
                continue

        pixel_idx = int(r) * 5 + int(c)
        primary_pixels.append(pixel_idx)

    return primary_pixels


def build_parentage_map(df: pd.DataFrame) -> dict:
    """Construit la carte trackID -> parentID pour tout le dataset."""
    if not {"trackID","parentID"}.issubset(df.columns):
        return {}
    s_tid = pd.to_numeric(df["trackID"], errors="coerce").dropna().astype(int)
    s_pid = pd.to_numeric(df["parentID"], errors="coerce").dropna().astype(int)
    # aligner par index commun
    s_pid = s_pid.reindex(s_tid.index)
    return dict(zip(s_tid.values, s_pid.values))


def find_primary_trackID(track_id, track_to_parent, max_depth=100):
    """Remonte la chaîne de parenté jusqu'au primaire."""
    current_track = track_id
    depth = 0
    visited = set()
    while depth < max_depth:
        if current_track in visited:
            return None  # cycle
        visited.add(current_track)
        parent = track_to_parent.get(current_track, None)
        if parent is None:
            return None
        if parent == 0:
            return current_track  # C'est le primaire !
        current_track = parent
        depth += 1
    return None



def identify_primary_pixels_fallback(g: pd.DataFrame, 
                                    energy_frac_range=(0.28, 0.48)) -> List[int]:
    """
    Fallback si parentID non disponible : temps + énergie + flags scatter==0.
    """
    g_no_scatter = g[(g.get("nCrystalCompton", 0) == 0) & (g.get("nCrystalRayleigh", 0) == 0)]
    if len(g_no_scatter) == 0:
        return []

    total_E = g["edep"].sum()
    primary_pixels = []

    for (r, c), grp in g_no_scatter.groupby(["row", "col"]):
        pixel_E = grp["edep"].sum()
        frac = pixel_E / total_E
        if energy_frac_range[0] <= frac <= energy_frac_range[1]:
            pixel_idx = int(r) * 5 + int(c)
            primary_pixels.append(pixel_idx)

    if len(primary_pixels) == 0:
        t_min_by_pix = g_no_scatter.groupby(["row", "col"])["time_ns"].min()
        if len(t_min_by_pix) > 0:
            r, c = t_min_by_pix.idxmin()
            primary_pixels.append(int(r) * 5 + int(c))

    return primary_pixels

# def identify_primary_pixels_v8(g, verbose=False):
#     """
#     Identifie les pixels primaires dans un pulse ICS.
    
#     NOUVELLE VERSION:
#     - Un primaire = scatter avec temps minimal PAR eventID
#     - Un pulse peut avoir PLUSIEURS primaires (un par eventID)
    
#     Args:
#         g: DataFrame groupé par pulse (tous les hits du pulse)
#         verbose: Afficher les détails
    
#     Returns:
#         Liste des indices de pixels primaires [0-24]
#     """
#     from collections import defaultdict
    
#     # Trouver tous les pixels avec scatter
#     pixels_with_scatter = [
#         p for p in g if (p.get('nCrystalCompton', 0) > 0 or p.get('nCrystalRayleigh', 0) > 0)
#     ]
    
#     if not pixels_with_scatter:
#         return []
    
#     #  NOUVEAU : Grouper par eventID
#     events = defaultdict(list)
#     for p in pixels_with_scatter:
#         event_id = p.get('eventID', 0)
#         events[event_id].append(p)
    
#     #  NOUVEAU : Trouver le primaire de CHAQUE eventID
#     primaries = []
#     for event_id, hits in events.items():
#         # Primaire = hit avec temps minimal dans cet eventID
#         # Utiliser 'time_ns' au lieu de 'time'
#         primary = min(hits, key=lambda p: p.get('time_ns', p.get('time', 0)))
#         pixel_idx = primary['row'] * 5 + primary['col']
#         primaries.append(pixel_idx)
        
#         if verbose:
#             print(f"   >>> EventID {event_id}: PRIMAIRE pixel ({primary['row']},{primary['col']}) "
#                   f"idx={pixel_idx}, t={primary.get('time_ns', primary.get('time', 0)):.2f} ns, "
#                   f"nHits={len(hits)}")
    
#     if verbose and len(primaries) > 1:
#         print(f"   >>> TOTAL: {len(primaries)} primaires détectés dans ce pulse")
    
#     return primaries
    
def identify_primary_pixels_v8(g: pd.DataFrame, verbose: bool = False) -> List[int]:
    """
    Nouvelle implémentation (logique v8) avec support de PLUSIEURS primaires par pulse.
    - On détecte, pour chaque eventID, le hit scatter au temps minimal (primaire v8).
    - On retourne ensuite TOUS ces pixels primaires (un ou plusieurs par pulse).
    - Retour: [] si pas d'ICS, sinon liste de pixel_idx (0..24).
    """
    # Colonnes éventuellement manquantes -> séries nulles par défaut
    comp = g["nCrystalCompton"] if "nCrystalCompton" in g.columns else pd.Series(0, index=g.index)
    rayl = g["nCrystalRayleigh"] if "nCrystalRayleigh" in g.columns else pd.Series(0, index=g.index)
    # Nouveau : support des données labellisées avec une colonne binaire is_scatter
    if "is_scatter" in g.columns:
        scat = pd.to_numeric(g["is_scatter"], errors="coerce").fillna(0)
    else:
        scat = pd.Series(0, index=g.index)

    # Y a-t-il au moins un scatter dans le pulse ?
    has_scatter = ((comp > 0) | (rayl > 0) | (scat > 0)).any()
    if not has_scatter:
        return []  # Pas un pulse ICS

    # Préférer 'time_ns', sinon fallback sur 'time'
    time_col = "time_ns" if "time_ns" in g.columns else ("time" if "time" in g.columns else None)
    if time_col is None:
        # Impossible d'appliquer la logique sans temps -> garder la compatibilité (pas de primaire)
        if verbose:
            print("   Aucun time/time_ns disponible -> []")
        return []

    # eventID: si absent, on considère un seul eventID=0 pour tout le pulse
    if "eventID" in g.columns:
        event_series = g["eventID"]
    else:
        event_series = pd.Series(0, index=g.index)

    # Restreindre aux hits scatter
    mask_scatter = (comp > 0) | (rayl > 0) | (scat > 0)
    g_sc = g.loc[mask_scatter].copy()
    if g_sc.empty:
        if verbose:
            print("   Aucun pixel avec scatter trouvé")
        return []

    # Vérifs colonnes (row/col) nécessaires pour l'index 0..24
    if not {"row", "col"}.issubset(g_sc.columns):
        if verbose:
            print("   Colonnes 'row'/'col' manquantes -> []")
        return []

    # Injecter un champ event utilisé pour le groupby
    g_sc["_event"] = event_series.loc[g_sc.index].values

    # Trouver le primaire par eventID: hit au temps minimal
    primaries = []  # list of dict(row, col, time, idx)
    for ev_id, df_ev in g_sc.groupby("_event"):
        # idx du temps minimal dans cet event
        try:
            i_min = df_ev[time_col].idxmin()
        except ValueError:
            continue  # sécurité si df_ev vide
        hit = df_ev.loc[i_min]
        r, c = int(hit["row"]), int(hit["col"])
        t = float(hit[time_col])
        pixel_idx = r * 5 + c
        primaries.append({"event": ev_id, "row": r, "col": c, "time": t, "idx": pixel_idx})

        if verbose:
            print(f"   >>> EventID {ev_id}: PRIMAIRE pixel ({r},{c}) "
                  f"idx={pixel_idx}, t={t:.2f} ns, nHits={len(df_ev)}")

    if not primaries:
        return []

    # Retourner TOUS les pixels primaires détectés (un ou plusieurs par pulse).
    # On enlève les doublons éventuels en cas de même pixel pour plusieurs events.
    primary_indices = sorted({int(p["idx"]) for p in primaries})

    if verbose:
        print(f"   >>> TOTAL: {len(primary_indices)} primaires (v8) détectés dans ce pulse:")
        for p in primaries:
            print(f"       - pixel ({p['row']},{p['col']}) idx={p['idx']}, t={p['time']:.2f} ns")

    return primary_indices

def identify_primary_pixels_v9(g: pd.DataFrame, verbose=False) -> List[int]:
    """
    Label le pixel primaire = première interaction scatter (temps minimal).
    """
    # Vérifier que le pulse a du scatter QUELQUE PART
    has_scatter = ((g.get('nCrystalCompton', 0) > 0).any() or 
                   (g.get('nCrystalRayleigh', 0) > 0).any())
    
    if not has_scatter:
        return []  # Pas un pulse ICS
    
    if verbose:
        print(f"\n PULSE ICS - {len(g)} hits, E_total={g['edep'].sum():.3f}")
    
    # Identifier les pixels AVEC scatter (candidats primaires)
    pixels_with_scatter = []
    total_E = g['edep'].sum()
    
    for (r, c), grp_pix in g.groupby(["row", "col"]):
        has_compton = (grp_pix['nCrystalCompton'] > 0).any()
        has_rayleigh = (grp_pix['nCrystalRayleigh'] > 0).any()
        
        if has_compton or has_rayleigh:
            # Pixel avec scatter - candidat primaire
            t_min = grp_pix['time_ns'].min()
            E_tot = grp_pix['edep'].sum()
            frac = E_tot / total_E
            
            pixels_with_scatter.append({
                'row': r,
                'col': c,
                'time': t_min,
                'energy': E_tot,
                'frac': frac,
                'compton': has_compton,
                'rayleigh': has_rayleigh
            })
            
            if verbose:
                scatter_type = "C" if has_compton else "R"
                print(f"   Candidat: pixel ({r},{c}), t={t_min:.2f}, "
                      f"E={E_tot:.3f} ({frac:.1%}), type={scatter_type}")
    
    if len(pixels_with_scatter) == 0:
        if verbose:
            print("   Aucun pixel avec scatter trouvé")
        return []
    
    # Le primaire = celui avec le temps minimal parmi les pixels scatter
    primary = min(pixels_with_scatter, key=lambda p: p['time'])
    
    # Optionnel : vérifier que c'est un dépôt partiel (< 95% de l'énergie)
    if primary['frac'] > 0.95:
        if verbose:
            print(f"    Pixel ({primary['row']},{primary['col']}) "
                  f"dépose {primary['frac']:.1%} - photoélectrique?")
        # Vous pouvez choisir de le rejeter ou de le garder
        # return []  # Décommentez pour rejeter
    
    pixel_idx = int(primary['row']) * 5 + int(primary['col'])
    
    if verbose:
        print(f"   >>> PRIMAIRE: pixel ({primary['row']},{primary['col']}) "
              f"idx={pixel_idx}, t={primary['time']:.2f}, E={primary['frac']:.1%}")
    
    return [pixel_idx]

def identify_scatter_pixels(g: pd.DataFrame, verbose=False) -> List[int]:
    """
    Identifie TOUS les pixels contenant des hits scatter (Compton ou Rayleigh).
    
    Un pixel est un scatter si au moins un hit a nCrystalCompton > 0 OU nCrystalRayleigh > 0.
    
    Returns:
        Liste des indices de pixels (0-24) contenant au moins un scatter.
    """
    scatter_pixels = []
    
    for (r, c), grp_pix in g.groupby(["row", "col"]):
        has_compton = (grp_pix.get('nCrystalCompton', 0) > 0).any()
        has_rayleigh = (grp_pix.get('nCrystalRayleigh', 0) > 0).any()
        # Nouveau : support de la colonne is_scatter (binaire)
        if 'is_scatter' in grp_pix.columns:
            has_is_scatter = (pd.to_numeric(grp_pix['is_scatter'], errors='coerce').fillna(0) > 0).any()
        else:
            has_is_scatter = False
        
        if has_compton or has_rayleigh or has_is_scatter:
            pixel_idx = int(r) * 5 + int(c)
            scatter_pixels.append(pixel_idx)
            
            if verbose:
                E_tot = grp_pix['edep'].sum()
                scatter_type = []
                if has_compton:
                    scatter_type.append("Compton")
                if has_rayleigh:
                    scatter_type.append("Rayleigh")
                if has_is_scatter:
                    scatter_type.append("is_scatter")
                print(f"   Scatter: pixel ({r},{c}) idx={pixel_idx}, "
                      f"E={E_tot:.3f}, type={'+'.join(scatter_type)}")
    
    return scatter_pixels

# def aggregate_to_5x5(df_rc: pd.DataFrame,
#                      set_primary_only_if_ics: bool = True,
#                      max_debug_prints: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List]:
#     """Version avec nouvelle logique v8."""
    
#     E_list, T_list, isics_list, prim_list = [], [], [], []
#     debug_count = 0
    
#     for pid, g in df_rc.groupby("pulse_id"):
#         # Grilles
#         gridE = np.zeros((5,5), np.float32)
#         gridT = np.zeros((5,5), np.float32)
        
#         for (r,c), grp in g.groupby(["row","col"]):
#             gridE[int(r), int(c)] = grp["edep"].sum()
#             gridT[int(r), int(c)] = grp["time_ns"].min()
        
#         # ICS flag
#         nC = (g.get("nCrystalCompton", 0) > 0).any()
#         nR = (g.get("nCrystalRayleigh", 0) > 0).any()
#         is_ics = bool(nC or nR)
        
#         # Labeling primaires
#         primary_pixels = []
#         if is_ics and set_primary_only_if_ics:
#             verbose = (debug_count < max_debug_prints)
#             if verbose:
#                 print(f"\n{'#'*70}")
#                 print(f"# PULSE ID: {pid} (ICS #{debug_count+1})")
#                 print(f"{'#'*70}")
            
#             primary_pixels = identify_primary_pixels_v8(g, verbose=verbose)
            
#             if verbose:
#                 debug_count += 1
        
#         E_list.append(gridE)
#         T_list.append(gridT)
#         isics_list.append(is_ics)
#         prim_list.append(primary_pixels)
    
#     return np.stack(E_list), np.stack(T_list), np.array(isics_list, bool), prim_list

def aggregate_to_5x5(df_rc: pd.DataFrame,
                     set_primary_only_if_ics: bool = True,
                     max_debug_prints: int = 10) -> dict:
    """
    Agrège en grilles 5x5 avec canaux multiples, identifie primaires, scatters,
    assignments (primary→secondaries) et énergie corrigée.

    Returns:
        dict with keys:
            E:  (N, 5, 5)  total energy per pixel
            T:  (N, 5, 5)  min time per pixel
            N_hits: (N, 5, 5)  number of hits per pixel
            E_var:  (N, 5, 5)  energy variance per pixel
            E_min:  (N, 5, 5)  min single-hit energy per pixel
            is_ics: (N,) bool
            primary_pixels: list of N lists
            scatter_pixels: list of N lists
            assign_map: (N, 5, 5) int  — for each pixel, index into primary_pixels
                        (-1 = not scatter, 0..K-1 = assigned to k-th primary)
            E_corrected: (N, 5, 5) energy with secondary scatter deposits removed
            pulse_ids: (N,) int
    """

    E_list, T_list, Nhits_list, Evar_list, Emin_list = [], [], [], [], []
    isics_list, prim_list, scatter_list = [], [], []
    assign_list, Ecorr_list, pid_list = [], [], []
    debug_count = 0

    for pid, g in df_rc.groupby("pulse_id"):
        # --- Per-pixel multi-channel grids ---
        gridE    = np.zeros((5, 5), np.float32)
        gridT    = np.zeros((5, 5), np.float32)
        gridN    = np.zeros((5, 5), np.float32)
        gridEvar = np.zeros((5, 5), np.float32)
        gridEmin = np.full((5, 5), 0.0, dtype=np.float32)

        for (r, c), grp in g.groupby(["row", "col"]):
            ri, ci = int(r), int(c)
            energies = grp["edep"].values
            gridE[ri, ci] = energies.sum()
            gridT[ri, ci] = grp["time_ns"].min()
            gridN[ri, ci] = len(energies)
            gridEvar[ri, ci] = energies.var() if len(energies) > 1 else 0.0
            gridEmin[ri, ci] = energies.min() if len(energies) > 0 else 0.0

        # --- ICS flag ---
        nC = (g.get("nCrystalCompton", 0) > 0).any()
        nR = (g.get("nCrystalRayleigh", 0) > 0).any()
        if "is_scatter" in g.columns:
            nS = (pd.to_numeric(g["is_scatter"], errors="coerce").fillna(0) > 0).any()
        else:
            nS = False
        is_ics = bool(nC or nR or nS)

        # --- Labeling ---
        verbose = (debug_count < max_debug_prints and is_ics)
        if verbose:
            print(f"\n{'#'*70}")
            print(f"# PULSE ID: {pid} (ICS #{debug_count+1})")
            print(f"{'#'*70}")

        scatter_pixels = identify_scatter_pixels(g, verbose=verbose)

        primary_pixels = []
        if is_ics and set_primary_only_if_ics:
            primary_pixels = identify_primary_pixels_v8(g, verbose=verbose)

        if verbose:
            debug_count += 1

        # --- Assignment map & corrected energy ---
        assign_map = np.full((5, 5), -1, dtype=np.int32)

        # Corrected energy: only keep energy from non-scatter events
        # For each pixel, sum energy only from hits that are NOT scatter
        gridEcorr = np.zeros((5, 5), np.float32)
        for (r, c), grp in g.groupby(["row", "col"]):
            ri, ci = int(r), int(c)
            # Non-scatter hits only
            comp = grp["nCrystalCompton"] if "nCrystalCompton" in grp.columns else 0
            rayl = grp["nCrystalRayleigh"] if "nCrystalRayleigh" in grp.columns else 0
            non_scatter_mask = ~((comp > 0) | (rayl > 0))
            gridEcorr[ri, ci] = grp.loc[non_scatter_mask, "edep"].sum()

        if is_ics and len(primary_pixels) > 0:
            # Build assignment: for each scatter pixel, find the nearest primary
            prim_coords = np.array([(p // 5, p % 5) for p in primary_pixels], dtype=np.float32)
            for sp in scatter_pixels:
                sr, sc_col = sp // 5, sp % 5
                dists = np.sqrt((prim_coords[:, 0] - sr)**2 + (prim_coords[:, 1] - sc_col)**2)
                nearest_k = int(np.argmin(dists))
                assign_map[sr, sc_col] = nearest_k

        E_list.append(gridE)
        T_list.append(gridT)
        Nhits_list.append(gridN)
        Evar_list.append(gridEvar)
        Emin_list.append(gridEmin)
        isics_list.append(is_ics)
        prim_list.append(primary_pixels)
        scatter_list.append(scatter_pixels)
        assign_list.append(assign_map)
        Ecorr_list.append(gridEcorr)
        pid_list.append(int(pid))

    return {
        "E": np.stack(E_list),
        "T": np.stack(T_list),
        "N_hits": np.stack(Nhits_list),
        "E_var": np.stack(Evar_list),
        "E_min": np.stack(Emin_list),
        "is_ics": np.array(isics_list, dtype=bool),
        "primary_pixels": prim_list,
        "scatter_pixels": scatter_list,
        "assign_map": np.stack(assign_list),
        "E_corrected": np.stack(Ecorr_list),
        "pulse_ids": np.array(pid_list, dtype=int),
    }


def aggregate_by_event(df_rc: pd.DataFrame,
                       set_primary_only_if_ics: bool = True) -> Tuple[list, list, list, list, list, list]:
    """
    Agrège les hits par (pulse_id, eventID) — version entièrement vectorisée.

    Chaque sample = un événement = un cluster de pixels dans la grille 5x5.
    Le primaire est TOUJOURS un pixel unique = le scatter avec le temps minimal.

    Returns:
        E_list, T_list, isics_list, prim_list, scatter_list, meta_list
    """
    df = df_rc.copy()
    df["row"] = df["row"].astype(int).clip(0, 4)
    df["col"] = df["col"].astype(int).clip(0, 4)
    df["pixel_idx"] = df["row"] * 5 + df["col"]

    # scatter flag per hit
    sc_flag = pd.Series(False, index=df.index)
    if "nCrystalCompton"  in df.columns: sc_flag |= df["nCrystalCompton"]  > 0
    if "nCrystalRayleigh" in df.columns: sc_flag |= df["nCrystalRayleigh"] > 0
    if "is_scatter"       in df.columns:
        sc_flag |= pd.to_numeric(df["is_scatter"], errors="coerce").fillna(0) > 0
    df["_sc"] = sc_flag.astype(np.int8)

    # ── Step 1: pixel-level aggregation ──────────────────────────────────
    pix_agg = (df.groupby(["pulse_id", "eventID", "pixel_idx"], sort=True)
                 .agg(E=("edep", "sum"), T=("time_ns", "min"), sc=("_sc", "max"))
                 .reset_index())

    # ── Step 2: assign a dense integer index to each (pulse_id, eventID) ─
    ev_keys = pix_agg[["pulse_id", "eventID"]].drop_duplicates().reset_index(drop=True)
    ev_keys["ev_i"] = np.arange(len(ev_keys))
    N = len(ev_keys)

    pix_agg = pix_agg.merge(ev_keys, on=["pulse_id", "eventID"], how="left")

    ev_i = pix_agg["ev_i"].values.astype(int)
    px   = pix_agg["pixel_idx"].values.astype(int)

    # ── Step 3: build E/T grids with pure numpy indexing ─────────────────
    E_flat = np.zeros((N, 25), dtype=np.float32)
    T_flat = np.zeros((N, 25), dtype=np.float32)
    E_flat[ev_i, px] = pix_agg["E"].values.astype(np.float32)
    T_flat[ev_i, px] = pix_agg["T"].values.astype(np.float32)

    E_arr = E_flat.reshape(N, 5, 5)
    T_arr = T_flat.reshape(N, 5, 5)

    # ── Step 4: ICS flag per event ────────────────────────────────────────
    ics_flat = np.zeros(N, dtype=bool)
    sc_vals  = pix_agg["sc"].values.astype(bool)
    np.maximum.at(ics_flat.view(np.uint8), ev_i, sc_vals.view(np.uint8))

    # ── Step 5: scatter pixel lists per event ────────────────────────────
    sc_pix = pix_agg[pix_agg["sc"] > 0][["ev_i", "pixel_idx"]]
    scatter_lists = [[] for _ in range(N)]
    for ei, pi in zip(sc_pix["ev_i"].values, sc_pix["pixel_idx"].values):
        scatter_lists[int(ei)].append(int(pi))

    # ── Step 6: primary pixel per event (scatter with min time) ──────────
    prim_lists = [[] for _ in range(N)]
    if set_primary_only_if_ics and len(sc_pix) > 0:
        sc_full = pix_agg[pix_agg["sc"] > 0][["ev_i", "pixel_idx", "T"]].copy()
        prim_rows = sc_full.loc[sc_full.groupby("ev_i")["T"].idxmin()]
        for ei, pi in zip(prim_rows["ev_i"].values, prim_rows["pixel_idx"].values):
            if ics_flat[int(ei)]:
                prim_lists[int(ei)] = [int(pi)]

    # ── Step 7: meta ──────────────────────────────────────────────────────
    meta_list = [
        {"pulse_id": int(r["pulse_id"]), "event_id": int(r["eventID"])}
        for _, r in ev_keys.iterrows()
    ]

    return (list(E_arr), list(T_arr), ics_flat.tolist(),
            prim_lists, scatter_lists, meta_list)


def aggregate_by_event_slow(df_rc: pd.DataFrame,
                       set_primary_only_if_ics: bool = True) -> Tuple[list, list, list, list, list, list]:
    """
    Agrège les hits par (pulse_id, eventID) au lieu de pulse_id seul.

    Chaque sample = un événement = un cluster de pixels dans la grille 5x5.
    Le primaire est TOUJOURS un pixel unique = le scatter avec le temps minimal
    dans cet événement (plus d'ambiguïté multi-primaires).

    Returns:
        E_list      : liste de (5,5) float32 — énergie de l'event
        T_list      : liste de (5,5) float32 — temps min par pixel de l'event
        isics_list  : liste de bool — l'event a-t-il du scatter ?
        prim_list   : liste de listes [pixel_idx] (longueur 0 ou 1)
        scatter_list: liste de listes [pixel_idx, ...] — scatters de l'event
        meta_list   : liste de dicts {"pulse_id": int, "event_id": int}
    """
    E_list, T_list, isics_list, prim_list, scatter_list, meta_list = [], [], [], [], [], []

    for (pid, ev_id), g_ev in df_rc.groupby(["pulse_id", "eventID"]):
        gridE = np.zeros((5, 5), dtype=np.float32)
        gridT = np.zeros((5, 5), dtype=np.float32)

        for (r, c), grp in g_ev.groupby(["row", "col"]):
            gridE[int(r), int(c)] = grp["edep"].sum()
            gridT[int(r), int(c)] = grp["time_ns"].min()

        # ICS flag pour cet event
        nC = (g_ev["nCrystalCompton"] > 0).any() if "nCrystalCompton" in g_ev.columns else False
        nR = (g_ev["nCrystalRayleigh"] > 0).any() if "nCrystalRayleigh" in g_ev.columns else False
        nS = False
        if "is_scatter" in g_ev.columns:
            nS = (pd.to_numeric(g_ev["is_scatter"], errors="coerce").fillna(0) > 0).any()
        is_ics = bool(nC or nR or nS)

        # Tous les pixels scatter de cet event
        scatter_pixels = identify_scatter_pixels(g_ev)

        # Pixel primaire unique = scatter avec temps minimal dans cet event
        primary_pixels = []
        if is_ics and set_primary_only_if_ics and len(scatter_pixels) > 0:
            comp = g_ev["nCrystalCompton"] if "nCrystalCompton" in g_ev.columns else pd.Series(0, index=g_ev.index)
            rayl = g_ev["nCrystalRayleigh"] if "nCrystalRayleigh" in g_ev.columns else pd.Series(0, index=g_ev.index)
            scat = pd.Series(0, index=g_ev.index)
            if "is_scatter" in g_ev.columns:
                scat = pd.to_numeric(g_ev["is_scatter"], errors="coerce").fillna(0)

            mask_sc = (comp > 0) | (rayl > 0) | (scat > 0)
            g_sc = g_ev[mask_sc]

            if not g_sc.empty and {"row", "col"}.issubset(g_sc.columns):
                i_min = g_sc["time_ns"].idxmin()
                hit = g_sc.loc[i_min]
                r, c = int(hit["row"]), int(hit["col"])
                primary_pixels = [r * 5 + c]

        E_list.append(gridE)
        T_list.append(gridT)
        isics_list.append(is_ics)
        prim_list.append(primary_pixels)
        scatter_list.append(scatter_pixels)
        meta_list.append({"pulse_id": int(pid), "event_id": int(ev_id)})

    return E_list, T_list, isics_list, prim_list, scatter_list, meta_list
