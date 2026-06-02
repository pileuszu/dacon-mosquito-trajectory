import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
import pickle

sys.path.append(os.getcwd())
from step39_six_regime.physics import make_candidates, extract_multi_scale_derivatives, EPS

# CLUSTER_FEATURES used in GMM clustering
CLUSTER_FEATURES = [
    "ctx_speed",
    "ctx_acc",
    "smooth_curv_w5",
    "ctx_lat_accel",
    "ctx_z_vel",
    "ctx_z_acc",
    "ctx_p_saccade",
    "roll_speed_cv_all"
]

# Mapping GMM-6 clusters to unique regime names
REGIME_MAPPING = {
    0: "fast_straight_low",
    1: "slow_moderate_turning",
    2: "fast_moderate_turning",
    3: "fast_straight_high",
    4: "fast_extreme_turning",
    5: "slow_extreme_turning"
}

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

def prepare_step39_data():
    print("=== Step 39: Preparing 6-Regime Training Datasets ===")
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    
    # Load EqMotion predictions for train set
    print("Loading EqMotion train predictions...")
    s4_preds_df = pd.read_csv("experiments/step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df.index.unique()
    
    # Load GMM-6 components
    models_dir = Path("step39_six_regime/models")
    with open(models_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(models_dir / "gmm_model.pkl", "rb") as f:
        gmm = pickle.load(f)
        
    # Extract clustering features for all trajectories
    print("Extracting features and predicting GMM-6 cluster indices...")
    traj_features = []
    traj_ids = []
    
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        ctx = extract_context_features(xyz)
        
        feat_vector = [ctx[feat] for feat in CLUSTER_FEATURES]
        traj_features.append(feat_vector)
        traj_ids.append(fid)
        
    traj_features = np.array(traj_features, dtype=np.float32)
    features_scaled = scaler.transform(traj_features)
    clusters = gmm.predict(features_scaled)
    
    # Map GMM predicted cluster names
    traj_regimes = {traj_ids[i]: REGIME_MAPPING[clusters[i]] for i in range(len(traj_ids))}
    
    # 6-Regime datasets mapping
    regime_rows = {name: [] for name in REGIME_MAPPING.values()}
    
    TARGET_THRESHOLD = 0.01  # 1.0cm
    
    print("\nPreparing Step 39 Physics-Explicit 6-Regime Datasets...")
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)
        
        # Calculate Priors
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        regime = traj_regimes[fid]
        
        # Extract features and candidates
        ctx = extract_context_features(xyz)
        cands, cands_list, c_features = make_candidates(xyz, priors=priors, end_idx=-1, horizon=2, regime=regime)
        
        dists = np.linalg.norm(cands - target, axis=1)
        num_physical = len(cands_list) - 2
        
        # Candidate selection logic based on regime
        if regime == "slow_moderate_turning":
            # Compact grid (22 candidates): select all to prevent missing any data points
            selected_indices = list(range(len(cands_list)))
        else:
            # Expanded grids: sample balanced candidates
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
                "grid_scale": c_features["grid_scale"][idx],
                "cand_speed": c_features["cand_speed"][idx],
                "cand_speed_ratio": c_features["cand_speed_ratio"][idx],
                "cand_turn_angle": c_features["cand_turn_angle"][idx],
                "cand_turn_rate": c_features["cand_turn_rate"][idx],
                "cand_accel": c_features["cand_accel"][idx],
                "cand_lat_accel": c_features["cand_lat_accel"][idx],
                **ctx,
                "dist_to_p0": np.linalg.norm(cand_pos - p0),
                "dist_to_s7": np.linalg.norm(cand_pos - s7_pos),
                "dist_to_s4": np.linalg.norm(cand_pos - s4_pos),
                "target": 1 if dist <= TARGET_THRESHOLD else 0,
                "reg_target": dist
            }
            regime_rows[regime].append(row)
            
    # Save the datasets
    Path("step39_six_regime/data").mkdir(parents=True, exist_ok=True)
    for regime, rows in regime_rows.items():
        df_reg = pd.DataFrame(rows)
        save_path = Path(f"step39_six_regime/data/train_ranker_v39_{regime}.csv")
        df_reg.to_csv(save_path, index=False)
        pos_count = df_reg['target'].sum() if 'target' in df_reg.columns else 0
        print(f"Saved {regime} data to {save_path} ({len(df_reg)} rows, {pos_count} positives)")

if __name__ == "__main__":
    prepare_step39_data()
