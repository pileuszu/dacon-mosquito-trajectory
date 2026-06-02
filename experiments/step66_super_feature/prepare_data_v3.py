import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from step66_super_feature.feature_extractor import extract_super_features

def main():
    source_dir = Path("step65_spatiotemporal_ai/data")
    dest_dir = Path("step66_super_feature/data")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw Step 65 datasets...")
    train_x = np.load(source_dir / "train_x.npy")
    train_y = np.load(source_dir / "train_y.npy")
    train_candidates = np.load(source_dir / "train_candidates.npy")  # [10000, 36, 3]
    with open(source_dir / "train_ids.json", "r") as f:
        train_ids = json.load(f)
        
    test_x = np.load(source_dir / "test_x.npy")
    test_candidates = np.load(source_dir / "test_candidates.npy")    # [10000, 36, 3]
    with open(source_dir / "test_ids.json", "r") as f:
        test_ids = json.load(f)
        
    # Load CFM predictions (Step 65 models)
    print("Loading CFM predictions...")
    train_cfm_preds = np.load(source_dir / "oof_preds_cfm_2step.npy")   # [10000, 3]
    test_cfm_preds = np.load(source_dir / "test_preds_cfm_2step.npy")     # [10000, 3]
    
    # Extract super features for history contexts
    print("Extracting 180+ dimensional super features...")
    train_feat = extract_super_features(train_x)  # [10000, 128]
    test_feat = extract_super_features(test_x)    # [10000, 128]
    
    # ------------------ Build Fold ID & Kinematics vectors ------------------
    import hashlib
    def stable_fold_id(sample_id: str, folds: int = 5) -> int:
        digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % folds

    fold_ids_tr = [stable_fold_id(sid) for sid in train_ids]
    
    # Pre-calculate Frenet unit vectors
    def get_frenet_and_dynamics(x_data):
        EPS = 1e-8
        v = (x_data[:, 1:] - x_data[:, :-1]) / 0.04
        a = (v[:, 1:] - v[:, :-1]) / 0.04
        v_last = v[:, -1]
        a_last = a[:, -1]
        
        speeds = np.linalg.norm(v_last, axis=1)
        T = v_last / (speeds[:, None] + EPS)
        a_par = np.sum(a_last * T, axis=1)
        a_perp = a_last - a_par[:, None] * T
        a_perp_norm = np.linalg.norm(a_perp, axis=1)
        N = a_perp / (a_perp_norm[:, None] + EPS)
        
        fallback = np.zeros_like(N)
        axis = np.argmin(np.abs(T), axis=1)
        fallback[np.arange(N.shape[0]), axis] = 1.0
        fallback = fallback - np.sum(fallback * T, axis=1)[:, None] * T
        fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + EPS)
        N = np.where(a_perp_norm[:, None] > 1e-6, N, fallback)
        
        B = np.cross(T, N, axis=1)
        B = B / (np.linalg.norm(B, axis=1)[:, None] + EPS)
        
        # Curvature
        cross_prod = np.cross(v_last, a_last, axis=1)
        cross_norm = np.linalg.norm(cross_prod, axis=1)
        curvatures = cross_norm / (speeds ** 3 + EPS)
        
        return T, N, B, speeds, curvatures, a_perp_norm
    
    print("Calculating Frenet systems & flight regimes...")
    T_tr, N_tr, B_tr, speeds_tr, curv_tr, a_perp_tr = get_frenet_and_dynamics(train_x)
    T_te, N_te, B_te, speeds_te, curv_te, a_perp_te = get_frenet_and_dynamics(test_x)
    
    p_last_tr = train_x[:, -1]
    p_last_te = test_x[:, -1]
    
    # Classify Regimes to apply dynamic delta scaling
    def get_dynamic_delta(speed, curvature, acc_perp_norm):
        is_steering = (curvature > 6.0) | (acc_perp_norm > 1.8)
        if is_steering:
            delta = 0.012  # power=0.0
        elif speed <= 0.50:
            delta = 0.005 * (speed / 0.36)  # power=1.0
        else:
            delta = 0.008 * (speed / 0.80)  # power=1.0
        return np.clip(delta, 0.001, 0.025)
    
    # ------------------ Build Train Tabular Dataset (43 Candidates, Sub-sampled) ------------------
    print("Building training tabular DataFrame (with negative sub-sampling & 43 dynamic hybrid candidates)...")
    
    records = []
    num_train = len(train_x)
    train_candidates_hybrid = np.zeros((num_train, 43, 3))
    
    for idx in range(num_train):
        p_last = p_last_tr[idx]
        T_vec = T_tr[idx]
        N_vec = N_tr[idx]
        B_vec = B_tr[idx]
        
        speed = speeds_tr[idx]
        curv = curv_tr[idx]
        a_perp = a_perp_tr[idx]
        
        # Calculate dynamic delta
        delta = get_dynamic_delta(speed, curv, a_perp)
        
        cfm_coord = train_cfm_preds[idx]
        y_true = train_y[idx]
        cands_36 = train_candidates[idx]
        
        # Build 7 Extra CFM-guided candidates dynamically
        extra_cands = np.array([
            cfm_coord,
            cfm_coord + delta * T_vec,
            cfm_coord - delta * T_vec,
            cfm_coord + delta * N_vec,
            cfm_coord - delta * N_vec,
            cfm_coord + delta * B_vec,
            cfm_coord - delta * B_vec
        ])
        
        cands_43 = np.vstack([cands_36, extra_cands])
        train_candidates_hybrid[idx] = cands_43
        
        dists = np.linalg.norm(cands_43 - y_true, axis=1)
        best_cand_idx = np.argmin(dists)
        
        # Sub-sample: best candidate + 7 other candidates (Hard negative mining)
        other_indices = np.array([i for i in range(43) if i != best_cand_idx])
        other_dists = dists[other_indices] # in cm
        
        # Hard Negatives: distance to true target between 1.0cm and 2.5cm
        hard_mask = (other_dists > 1.0) & (other_dists <= 2.5)
        hard_indices = other_indices[hard_mask]
        
        # Easy Negatives: distance to true target > 2.5cm
        easy_mask = other_dists > 2.5
        easy_indices = other_indices[easy_mask]
        
        # Sample up to 3 hard negatives
        num_hard_to_sample = min(3, len(hard_indices))
        if num_hard_to_sample > 0:
            sampled_hard = np.random.choice(hard_indices, size=num_hard_to_sample, replace=False)
        else:
            sampled_hard = np.array([], dtype=int)
            
        # Sample easy negatives to fill up to 7
        num_easy_to_sample = 7 - len(sampled_hard)
        available_easy = easy_indices
        if len(available_easy) < num_easy_to_sample:
            remaining = np.array([i for i in other_indices if i not in sampled_hard])
            sampled_easy = np.random.choice(remaining, size=num_easy_to_sample, replace=False)
        else:
            sampled_easy = np.random.choice(available_easy, size=num_easy_to_sample, replace=False)
            
        selected_indices = np.concatenate([[best_cand_idx], sampled_hard, sampled_easy])
        
        for cand_idx in selected_indices:
            cand_coord = cands_43[cand_idx]
            disp = cand_coord - p_last
            
            # Project to Frenet frame
            spec_par = np.sum(disp * T_vec)
            spec_perp = np.sum(disp * N_vec)
            spec_ts = np.sum(disp * B_vec)
            
            # Distance from candidate to CFM prediction
            dist_to_cfm = np.linalg.norm(cand_coord - cfm_coord) * 100.0 # in cm
            dist_target = dists[cand_idx] * 100.0 # in cm
            
            row = {
                "sample_idx": idx,
                "cand_idx": cand_idx,
                "fold_id": fold_ids_tr[idx],
                "spec_par": spec_par,
                "spec_perp": spec_perp,
                "spec_ts": spec_ts,
                "dist_to_cfm": dist_to_cfm,
                "dist_target": dist_target
            }
            # Append history features
            for f_idx in range(train_feat.shape[1]):
                row[f"feat_{f_idx}"] = train_feat[idx, f_idx]
                
            records.append(row)
            
    train_df = pd.DataFrame(records)
    train_df.to_parquet(dest_dir / "train_tabular_v3.parquet", index=False)
    np.save(dest_dir / "train_candidates_hybrid_v3.npy", train_candidates_hybrid)
    print(f"  Saved train_tabular_v3.parquet with shape: {train_df.shape}")
    
    # ------------------ Build Full Train Tabular Dataset for True OOF Evaluation (Full 43 Candidates) ------------------
    print("Building full training tabular DataFrame for True OOF evaluation (all 43 dynamic candidates)...")
    full_records = []
    
    for idx in range(num_train):
        p_last = p_last_tr[idx]
        T_vec = T_tr[idx]
        N_vec = N_tr[idx]
        B_vec = B_tr[idx]
        
        cfm_coord = train_cfm_preds[idx]
        y_true = train_y[idx]
        cands_43 = train_candidates_hybrid[idx]
        dists = np.linalg.norm(cands_43 - y_true, axis=1)
        
        for cand_idx in range(43):
            cand_coord = cands_43[cand_idx]
            disp = cand_coord - p_last
            
            spec_par = np.sum(disp * T_vec)
            spec_perp = np.sum(disp * N_vec)
            spec_ts = np.sum(disp * B_vec)
            
            dist_to_cfm = np.linalg.norm(cand_coord - cfm_coord) * 100.0 # in cm
            dist_target = dists[cand_idx] * 100.0 # in cm
            
            row = {
                "sample_idx": idx,
                "cand_idx": cand_idx,
                "fold_id": fold_ids_tr[idx],
                "spec_par": spec_par,
                "spec_perp": spec_perp,
                "spec_ts": spec_ts,
                "dist_to_cfm": dist_to_cfm,
                "dist_target": dist_target
            }
            for f_idx in range(train_feat.shape[1]):
                row[f"feat_{f_idx}"] = train_feat[idx, f_idx]
                
            full_records.append(row)
            
    train_full_df = pd.DataFrame(full_records)
    train_full_df.to_parquet(dest_dir / "train_tabular_full_v3.parquet", index=False)
    print(f"  Saved train_tabular_full_v3.parquet with shape: {train_full_df.shape}")
    
    # ------------------ Build Test Tabular Dataset (Full 43 Candidates) ------------------
    print("Building test tabular DataFrame (full 43 dynamic candidates)...")
    test_records = []
    num_test = len(test_x)
    
    test_candidates_hybrid = np.zeros((num_test, 43, 3))
    
    for idx in range(num_test):
        p_last = p_last_te[idx]
        T_vec = T_te[idx]
        N_vec = N_te[idx]
        B_vec = B_te[idx]
        
        speed = speeds_te[idx]
        curv = curv_te[idx]
        a_perp = a_perp_te[idx]
        
        delta = get_dynamic_delta(speed, curv, a_perp)
        
        cfm_coord = test_cfm_preds[idx]
        cands_36 = test_candidates[idx]
        
        extra_cands = np.array([
            cfm_coord,
            cfm_coord + delta * T_vec,
            cfm_coord - delta * T_vec,
            cfm_coord + delta * N_vec,
            cfm_coord - delta * N_vec,
            cfm_coord + delta * B_vec,
            cfm_coord - delta * B_vec
        ])
        
        cands_43 = np.vstack([cands_36, extra_cands])
        test_candidates_hybrid[idx] = cands_43
        
        for cand_idx in range(43):
            cand_coord = cands_43[cand_idx]
            disp = cand_coord - p_last
            
            spec_par = np.sum(disp * T_vec)
            spec_perp = np.sum(disp * N_vec)
            spec_ts = np.sum(disp * B_vec)
            
            dist_to_cfm = np.linalg.norm(cand_coord - cfm_coord) * 100.0 # in cm
            
            row = {
                "sample_idx": idx,
                "cand_idx": cand_idx,
                "spec_par": spec_par,
                "spec_perp": spec_perp,
                "spec_ts": spec_ts,
                "dist_to_cfm": dist_to_cfm
            }
            for f_idx in range(test_feat.shape[1]):
                row[f"feat_{f_idx}"] = test_feat[idx, f_idx]
                
            test_records.append(row)
            
    test_df = pd.DataFrame(test_records)
    test_df.to_parquet(dest_dir / "test_tabular_v3.parquet", index=False)
    np.save(dest_dir / "test_candidates_hybrid_v3.npy", test_candidates_hybrid)
    print(f"  Saved test_tabular_v3.parquet with shape: {test_df.shape}")
    print("Step 67 (V3 Dynamic Grid) Data preparation complete.")

if __name__ == "__main__":
    main()
