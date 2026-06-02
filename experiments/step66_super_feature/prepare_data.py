import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from feature_extractor import extract_super_features

def main():
    data_dir = Path("step65_spatiotemporal_ai/data")
    dest_dir = Path("step66_super_feature/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw Step 65 datasets...")
    train_x = np.load(data_dir / "train_x.npy")
    train_y = np.load(data_dir / "train_y.npy")
    train_candidates = np.load(data_dir / "train_candidates.npy")  # [10000, 36, 3]
    with open(data_dir / "train_ids.json", "r") as f:
        train_ids = json.load(f)
        
    test_x = np.load(data_dir / "test_x.npy")
    test_candidates = np.load(data_dir / "test_candidates.npy")    # [10000, 36, 3]
    with open(data_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
        
    # Copy source files to dest_dir for reference
    np.save(dest_dir / "train_x.npy", train_x)
    np.save(dest_dir / "train_y.npy", train_y)
    np.save(dest_dir / "train_candidates.npy", train_candidates)
    with open(dest_dir / "train_ids.json", "w") as f:
        json.dump(train_ids, f)
    np.save(dest_dir / "test_x.npy", test_x)
    np.save(dest_dir / "test_candidates.npy", test_candidates)
    with open(dest_dir / "test_ids.json", "w") as f:
        json.dump(test_ids, f)
        
    # Extract super features for history contexts
    print("Extracting 180+ dimensional super features...")
    train_feat = extract_super_features(train_x)  # [10000, 182]
    test_feat = extract_super_features(test_x)    # [10000, 182]
    
    print(f"  Train features shape: {train_feat.shape}")
    print(f"  Test features shape: {test_feat.shape}")
    
    # ------------------ Build Train Tabular Dataset (with Sub-sampling) ------------------
    # To prevent AutoGluon memory blowup and speed up fit time, we sub-sample candidate rows:
    # For each sample, select the best candidate (smallest distance to y) and 7 other candidates at random/spread.
    print("Building training tabular DataFrame (with negative sub-sampling & fold_id)...")
    
    import hashlib
    def stable_fold_id(sample_id: str, folds: int = 5) -> int:
        digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % folds

    fold_ids_tr = [stable_fold_id(sid) for sid in train_ids]
    
    records = []
    num_train = len(train_x)
    
    # Pre-calculate candidate physical specs (offsets relative to p_last)
    # spec_par, spec_perp, spec_ts (x, y, z offsets in Frenet Frame relative to p_last)
    # In past steps, train_candidates is already in absolute Cartesian coordinates.
    # To construct the unscaled specs, we project (candidates - p_last) onto the Frenet Frame:
    # speed_last, T, N, B
    v_tr = (train_x[:, 1:] - train_x[:, :-1]) / 0.04
    a_tr = (v_tr[:, 1:] - v_tr[:, :-1]) / 0.04
    v_last_tr = v_tr[:, -1]
    a_last_tr = a_tr[:, -1]
    
    speeds_last_tr = np.linalg.norm(v_last_tr, axis=1)
    T_tr = v_last_tr / (speeds_last_tr[:, None] + 1e-8)
    a_par_tr = np.sum(a_last_tr * T_tr, axis=1)
    a_perp_tr = a_last_tr - a_par_tr[:, None] * T_tr
    a_perp_norm_tr = np.linalg.norm(a_perp_tr, axis=1)
    N_tr = a_perp_tr / (a_perp_norm_tr[:, None] + 1e-8)
    
    # Fallback for N
    fallback_tr = np.zeros_like(N_tr)
    axis_tr = np.argmin(np.abs(T_tr), axis=1)
    fallback_tr[np.arange(N_tr.shape[0]), axis_tr] = 1.0
    fallback_tr = fallback_tr - np.sum(fallback_tr * T_tr, axis=1)[:, None] * T_tr
    fallback_tr = fallback_tr / (np.linalg.norm(fallback_tr, axis=1)[:, None] + 1e-8)
    N_tr = np.where(a_perp_norm_tr[:, None] > 1e-6, N_tr, fallback_tr)
    
    B_tr = np.cross(T_tr, N_tr, axis=1)
    B_tr = B_tr / (np.linalg.norm(B_tr, axis=1)[:, None] + 1e-8)
    
    p_last_tr = train_x[:, -1]
    
    for idx in range(num_train):
        p_last = p_last_tr[idx]
        T = T_tr[idx]
        N = N_tr[idx]
        B = B_tr[idx]
        
        y_true = train_y[idx]
        cands = train_candidates[idx]  # [36, 3]
        
        dists = np.linalg.norm(cands - y_true, axis=1)
        best_cand_idx = np.argmin(dists)
        
        # Sub-sample indices: best candidate + 7 other random candidates
        other_indices = [i for i in range(36) if i != best_cand_idx]
        sampled_others = np.random.choice(other_indices, size=7, replace=False)
        selected_indices = np.append(best_cand_idx, sampled_others)
        
        for cand_idx in selected_indices:
            cand_coord = cands[cand_idx]
            disp = cand_coord - p_last
            
            # Project displacement to unscaled Frenet specs
            spec_par = np.sum(disp * T)
            spec_perp = np.sum(disp * N)
            spec_ts = np.sum(disp * B)
            
            # Target distance in cm
            dist_target = dists[cand_idx] * 100.0  # in cm
            
            # Construct a row dictionary
            row = {
                "sample_idx": idx,
                "cand_idx": cand_idx,
                "fold_id": fold_ids_tr[idx],
                "spec_par": spec_par,
                "spec_perp": spec_perp,
                "spec_ts": spec_ts,
                "dist_target": dist_target
            }
            # Append the 180+ history features
            for f_idx in range(train_feat.shape[1]):
                row[f"feat_{f_idx}"] = train_feat[idx, f_idx]
                
            records.append(row)
            
    train_df = pd.DataFrame(records)
    train_df.to_parquet(dest_dir / "train_tabular.parquet", index=False)
    print(f"  Saved train_tabular.parquet with shape: {train_df.shape}")
    
    # ------------------ Build Full Train Tabular Dataset for True OOF Evaluation (Full 36 Candidates) ------------------
    print("Building full training tabular DataFrame for True OOF evaluation (all 36 candidates)...")
    full_records = []
    
    for idx in range(num_train):
        p_last = p_last_tr[idx]
        T = T_tr[idx]
        N = N_tr[idx]
        B = B_tr[idx]
        
        y_true = train_y[idx]
        cands = train_candidates[idx]
        dists = np.linalg.norm(cands - y_true, axis=1)
        
        for cand_idx in range(36):
            cand_coord = cands[cand_idx]
            disp = cand_coord - p_last
            
            spec_par = np.sum(disp * T)
            spec_perp = np.sum(disp * N)
            spec_ts = np.sum(disp * B)
            
            dist_target = dists[cand_idx] * 100.0  # in cm
            
            row = {
                "sample_idx": idx,
                "cand_idx": cand_idx,
                "fold_id": fold_ids_tr[idx],
                "spec_par": spec_par,
                "spec_perp": spec_perp,
                "spec_ts": spec_ts,
                "dist_target": dist_target
            }
            for f_idx in range(train_feat.shape[1]):
                row[f"feat_{f_idx}"] = train_feat[idx, f_idx]
                
            full_records.append(row)
            
    train_full_df = pd.DataFrame(full_records)
    train_full_df.to_parquet(dest_dir / "train_tabular_full.parquet", index=False)
    print(f"  Saved train_tabular_full.parquet with shape: {train_full_df.shape}")
    
    # ------------------ Build Test Tabular Dataset (Full 36 Candidates) ------------------
    # For inference, we must evaluate ALL 36 candidates per sample
    print("Building test tabular DataFrame (full 36 candidates)...")
    test_records = []
    num_test = len(test_x)
    
    v_te = (test_x[:, 1:] - test_x[:, :-1]) / 0.04
    a_te = (v_te[:, 1:] - v_te[:, :-1]) / 0.04
    v_last_te = v_te[:, -1]
    a_last_te = a_te[:, -1]
    
    speeds_last_te = np.linalg.norm(v_last_te, axis=1)
    T_te = v_last_te / (speeds_last_te[:, None] + 1e-8)
    a_par_te = np.sum(a_last_te * T_te, axis=1)
    a_perp_te = a_last_te - a_par_te[:, None] * T_te
    a_perp_norm_te = np.linalg.norm(a_perp_te, axis=1)
    N_te = a_perp_te / (a_perp_norm_te[:, None] + 1e-8)
    
    fallback_te = np.zeros_like(N_te)
    axis_te = np.argmin(np.abs(T_te), axis=1)
    fallback_te[np.arange(N_te.shape[0]), axis_te] = 1.0
    fallback_te = fallback_te - np.sum(fallback_te * T_te, axis=1)[:, None] * T_te
    fallback_te = fallback_te / (np.linalg.norm(fallback_te, axis=1)[:, None] + 1e-8)
    N_te = np.where(a_perp_norm_te[:, None] > 1e-6, N_te, fallback_te)
    
    B_te = np.cross(T_te, N_te, axis=1)
    B_te = B_te / (np.linalg.norm(B_te, axis=1)[:, None] + 1e-8)
    
    p_last_te = test_x[:, -1]
    
    for idx in range(num_test):
        p_last = p_last_te[idx]
        T = T_te[idx]
        N = N_te[idx]
        B = B_te[idx]
        
        cands = test_candidates[idx]
        
        for cand_idx in range(36):
            cand_coord = cands[cand_idx]
            disp = cand_coord - p_last
            
            spec_par = np.sum(disp * T)
            spec_perp = np.sum(disp * N)
            spec_ts = np.sum(disp * B)
            
            row = {
                "sample_idx": idx,
                "cand_idx": cand_idx,
                "spec_par": spec_par,
                "spec_perp": spec_perp,
                "spec_ts": spec_ts
            }
            for f_idx in range(test_feat.shape[1]):
                row[f"feat_{f_idx}"] = test_feat[idx, f_idx]
                
            test_records.append(row)
            
    test_df = pd.DataFrame(test_records)
    test_df.to_parquet(dest_dir / "test_tabular.parquet", index=False)
    print(f"  Saved test_tabular.parquet with shape: {test_df.shape}")
    print("Data preparation complete.")

if __name__ == "__main__":
    main()
