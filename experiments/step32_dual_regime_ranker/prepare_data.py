import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from step32_dual_regime_ranker.physics import make_candidates, extract_multi_scale_derivatives, EPS

def extract_context_features(xyz):
    ctx = extract_multi_scale_derivatives(xyz)
    
    vel = np.diff(xyz, axis=0)
    acc = np.diff(vel, axis=0)
    last_vel = vel[-1]
    speed = np.linalg.norm(last_vel)
    prev_speed = np.linalg.norm(vel[-2])
    
    cross_va = np.cross(last_vel, acc[-1])
    curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
    cos_theta = np.sum(vel[-1] * vel[-2]) / (speed * prev_speed + EPS)
    
    ctx["ctx_curv"] = curv
    ctx["ctx_turn"] = cos_theta
    
    return ctx

def prepare_step32_data():
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # Load EqMotion predictions for train set
    print("Loading EqMotion train predictions...")
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    
    slow_rows = []
    fast_rows = []
    
    TARGET_THRESHOLD = 0.01  # 1.0cm
    
    print("Preparing Step 32 Dual-Regime Ranker Data...")
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        # Calculate Priors
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        speed = np.linalg.norm(last_vel)
        
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        # Extract features and candidates
        ctx = extract_context_features(xyz)
        cands, cands_list = make_candidates(xyz, priors=priors, end_idx=-1, horizon=2)
        
        dists = np.linalg.norm(cands - target, axis=1)
        num_physical = len(cands_list) - 2
        
        # Split behavior by speed regime
        if speed <= 0.0234:
            # Slow regime: Keep ALL candidates (high density) without downsampling
            selected_indices = list(range(len(cands_list)))
            
            for idx in selected_indices:
                spec = cands_list[idx]
                cand_pos = cands[idx]
                dist = dists[idx]
                
                is_prior_val = 1 if idx >= num_physical else 0
                
                row = {
                    "id": fid,
                    "cand_idx": idx,
                    "spec_par": spec.par,
                    "spec_perp": spec.perp,
                    "spec_ts": spec.time_scale,
                    "spec_dmp": spec.damping,
                    "spec_jerk": spec.jerk,
                    "is_prior": is_prior_val,
                    **ctx,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0),
                    "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                    "dist_to_s4": np.linalg.norm(cand_pos - s4_pos),
                    "target": 1 if dist <= TARGET_THRESHOLD else 0,
                    "reg_target": dist
                }
                slow_rows.append(row)
        else:
            # Fast regime: Balance candidates to prevent target dilution
            best_idx = np.argmin(dists)
            neg_indices = np.where(dists > TARGET_THRESHOLD)[0]
            
            if len(neg_indices) > 20:
                random_negs = np.random.choice(neg_indices, 20, replace=False)
            else:
                random_negs = neg_indices
                
            near_miss_indices = neg_indices[np.argsort(dists[neg_indices])[:8]]
            prior_indices = [len(cands_list) - 2, len(cands_list) - 1]
            
            selected_indices = list(set([best_idx]) | set(random_negs) | set(near_miss_indices) | set(prior_indices))
            
            for idx in selected_indices:
                spec = cands_list[idx]
                cand_pos = cands[idx]
                dist = dists[idx]
                
                is_prior_val = 1 if idx >= num_physical else 0
                
                row = {
                    "id": fid,
                    "cand_idx": idx,
                    "spec_par": spec.par,
                    "spec_perp": spec.perp,
                    "spec_ts": spec.time_scale,
                    "spec_dmp": spec.damping,
                    "spec_jerk": spec.jerk,
                    "is_prior": is_prior_val,
                    **ctx,
                    "dist_to_p0": np.linalg.norm(cand_pos - p0),
                    "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                    "dist_to_s4": np.linalg.norm(cand_pos - s4_pos),
                    "target": 1 if dist <= TARGET_THRESHOLD else 0,
                    "reg_target": dist
                }
                fast_rows.append(row)
                
    # Save Slow regime dataset
    slow_df = pd.DataFrame(slow_rows)
    slow_path = Path("step32_dual_regime_ranker/train_ranker_v32_slow.csv")
    slow_df.to_csv(slow_path, index=False)
    print(f"Slow regime data saved: {slow_path} ({len(slow_df)} rows, {slow_df['target'].sum()} positives)")
    
    # Save Fast regime dataset
    fast_df = pd.DataFrame(fast_rows)
    fast_path = Path("step32_dual_regime_ranker/train_ranker_v32_fast.csv")
    fast_df.to_csv(fast_path, index=False)
    print(f"Fast regime data saved: {fast_path} ({len(fast_df)} rows, {fast_df['target'].sum()} positives)")

if __name__ == "__main__":
    prepare_step32_data()
