import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

sys.path.append(os.getcwd())
from step35_four_regime.physics import make_candidates, extract_multi_scale_derivatives, EPS

# The 8 features used in clustering
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

def prepare_step35_data():
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    
    # Load EqMotion predictions for train set
    print("Loading EqMotion train predictions...")
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    sample_ids = labels_df['id'].unique()
    
    # First Pass: Extract clustering features for all 10k trajectories
    print("Extracting trajectory features for GMM clustering...")
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
    
    # Fit StandardScaler and GaussianMixture
    print("Fitting GMM (K=4) to partition flight regimes...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(traj_features)
    
    gmm = GaussianMixture(n_components=4, random_state=42, n_init=5)
    clusters = gmm.fit_predict(features_scaled)
    
    # Save the models
    models_dir = Path("step35_four_regime/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    with open(models_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(models_dir / "gmm_model.pkl", "wb") as f:
        pickle.dump(gmm, f)
        
    print("Scaler and GMM models saved successfully.")
    
    # Determine the mapping from cluster index to physical regime dynamically
    cluster_stats = []
    for c in range(4):
        mask = (clusters == c)
        # Using raw features for stats: features_raw has shape (10000, 8)
        cluster_stats.append({
            'cluster_idx': c,
            'speed': np.mean(traj_features[mask, 0]),  # ctx_speed
            'curv': np.mean(traj_features[mask, 2])    # smooth_curv_w5
        })
        
    # Mapping logic:
    # 1. Sort by speed: lowest speed is slow_straight
    sorted_by_speed = sorted(cluster_stats, key=lambda x: x['speed'])
    slow_straight_idx = sorted_by_speed[0]['cluster_idx']
    
    # 2. Sort by curvature: highest curvature is slow_extreme_turning
    sorted_by_curv = sorted(cluster_stats, key=lambda x: x['curv'])
    slow_extreme_turning_idx = sorted_by_curv[-1]['cluster_idx']
    
    # 3. The other two are fast_straight and fast_turning.
    # The one with lower curvature is fast_straight, other is fast_turning.
    remaining = [x for x in cluster_stats if x['cluster_idx'] not in [slow_straight_idx, slow_extreme_turning_idx]]
    if remaining[0]['curv'] < remaining[1]['curv']:
        fast_straight_idx = remaining[0]['cluster_idx']
        fast_turning_idx = remaining[1]['cluster_idx']
    else:
        fast_straight_idx = remaining[1]['cluster_idx']
        fast_turning_idx = remaining[0]['cluster_idx']
        
    mapping = {
        slow_straight_idx: "slow_straight",
        slow_extreme_turning_idx: "slow_extreme_turning",
        fast_straight_idx: "fast_straight",
        fast_turning_idx: "fast_turning"
    }
    
    # Save the mapping config
    with open(models_dir / "regime_mapping.pkl", "wb") as f:
        pickle.dump(mapping, f)
        
    print("\nDynamic GMM Cluster Mapping:")
    for idx, regime in mapping.items():
        stats = next(x for x in cluster_stats if x['cluster_idx'] == idx)
        print(f"Cluster {idx} -> {regime} (Avg Speed: {stats['speed']*100:.2f} cm/s, Avg Curvature: {stats['curv']:.2f})")
        
    # Map trajectory IDs to their GMM regime
    traj_regimes = {traj_ids[i]: mapping[clusters[i]] for i in range(len(traj_ids))}
    
    # Second Pass: Prepare ranker datasets for each regime
    regime_rows = {
        "slow_straight": [],
        "slow_extreme_turning": [],
        "fast_straight": [],
        "fast_turning": []
    }
    
    TARGET_THRESHOLD = 0.01  # 1.0cm
    
    print("\nPreparing Step 35 Physics-Explicit Ranker Datasets...")
    for fid in tqdm(sample_ids):
        fpath = train_dir / f"{fid}.csv"
        df = pd.read_csv(fpath)
        xyz = df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
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
        
        if regime == "slow_straight":
            # Slow Cruising Regime: Keep ALL candidates (high density) without downsampling
            selected_indices = list(range(len(cands_list)))
        else:
            # Other Regimes: Balance candidates to prevent target dilution
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
    for regime, rows in regime_rows.items():
        df_reg = pd.DataFrame(rows)
        save_path = Path(f"step35_four_regime/train_ranker_v35_{regime}.csv")
        df_reg.to_csv(save_path, index=False)
        print(f"Saved {regime} data to {save_path} ({len(df_reg)} rows, {df_reg['target'].sum()} positives)")

if __name__ == "__main__":
    prepare_step35_data()
