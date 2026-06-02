import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from step44_regression_routing.physics import make_candidates, extract_multi_scale_derivatives, EPS
from step44_regression_routing.prepare_data import extract_context_features, REGIME_MAPPING

def search_best_weights(trajectory_data, true_targets, s4_priors, s7_priors, num_search_steps=20):
    best_hit = -1.0
    best_weights = None
    
    grid = []
    for i in range(num_search_steps + 1):
        w_m = i / num_search_steps
        for j in range(num_search_steps - i + 1):
            w_4 = j / num_search_steps
            w_7 = 1.0 - w_m - w_4
            grid.append((w_m, w_4, w_7))
            
    for weights in grid:
        w_m, w_4, w_7 = weights
        hits = []
        for fid, info in trajectory_data.items():
            final_coords = info['coords']
            s4_pos = s4_priors[fid]
            s7_pos = s7_priors[fid]
            target = true_targets[fid]
            
            blended = w_m * final_coords + w_4 * s4_pos + w_7 * s7_pos
            err = np.linalg.norm(blended - target)
            hits.append(1 if err <= 0.01 else 0)
            
        hit_rate = np.mean(hits)
        if hit_rate > best_hit:
            best_hit = hit_rate
            best_weights = weights
            
    return best_hit, best_weights

def main():
    print("=== Step 44: Optimizing Regression Blending Weights on Unbiased OOF Validation ===")
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    
    labels_df = pd.read_csv(data_dir / "train_labels.csv").set_index('id')
    s4_preds_df = pd.read_csv("experiments/step12/step4_preds_train.csv").set_index('id')
    
    print("Loading Step 44 OOF predictions...")
    oof_df = pd.read_csv("step44_regression_routing/data/oof_predictions_final.csv")
    oof_df = oof_df.sort_values(by=['id', 'cand_idx']).reset_index(drop=True)
    
    unique_ids = oof_df['id'].unique()
    print(f"Total Trajectories in OOF dataset: {len(unique_ids)}")
    
    true_targets = {}
    s4_priors = {}
    s7_priors = {}
    
    oof_grouped = {fid: gp for fid, gp in oof_df.groupby('id')}
    trajectory_data = {}
    
    print("Generating validation candidates and running anisotropic smoothing on regression scores...")
    for fid in tqdm(unique_ids):
        fpath = train_dir / f"{fid}.csv"
        with open(fpath, 'r') as f:
            lines = f.readlines()
        xyz = []
        for line in lines[1:]:
            parts = line.strip().split(',')
            xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        xyz = np.array(xyz, dtype=np.float32)
        
        target = labels_df.loc[fid].to_numpy(dtype=np.float32)
        true_targets[fid] = target
        
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        s4_priors[fid] = s4_pos
        s7_priors[fid] = s7_pos
        
        gp = oof_grouped[fid]
        regime = gp['regime'].values[0]
        gmm_cluster = gp['gmm_cluster'].values[0]
        
        cands, cands_list, c_features = make_candidates(xyz, priors=[s7_pos, s4_pos], end_idx=-1, horizon=2, regime=regime)
        
        cand_indices = gp['cand_idx'].values
        pred_dists = gp['pred_dist'].values
        
        # Convert distance to exponential score (higher is better)
        scores = np.exp(-pred_dists / 0.015)
        
        probs = np.zeros(len(cands), dtype=np.float32)
        for idx, s in zip(cand_indices, scores):
            if idx < len(probs):
                probs[idx] = s
                
        speed = np.linalg.norm(last_vel)
        speed_m_s = speed / 0.01
        
        ctx = extract_context_features(xyz)
        p_sacc = ctx["ctx_p_saccade"]
        
        S_grid = c_features["grid_scale"][0]
        S_scale = S_grid ** 1.5
        
        sigma_tangential = np.clip(0.0035 + 0.005 * speed_m_s * p_sacc, 0.003, 0.011) * S_scale
        sigma_normal = np.clip(0.0035 + 0.0015 * speed_m_s * p_sacc, 0.003, 0.006) * S_scale
        
        active_indices = np.where(probs > 1e-5)[0]
        if len(active_indices) == 0:
            active_indices = np.array([np.argmax(probs)])
            
        active_probs = probs[active_indices]
        
        tangent = last_vel / (speed + EPS)
        cands_diff = cands[:, None, :] - cands[None, active_indices, :]
        dx_tangential = np.dot(cands_diff, tangent)
        dx_sq = np.sum(cands_diff ** 2, axis=-1)
        dx_normal_sq = np.maximum(dx_sq - dx_tangential ** 2, 0.0)
        
        weights = np.exp(- (dx_tangential ** 2) / (2.0 * (sigma_tangential ** 2)) 
                          - dx_normal_sq / (2.0 * (sigma_normal ** 2)))
        
        smoothed_probs = weights.dot(active_probs)
        best_idx = np.argmax(smoothed_probs)
        final_coords = cands[best_idx]
        
        raw_err = np.linalg.norm(final_coords - target)
        
        trajectory_data[fid] = {
            'coords': final_coords,
            'gmm_cluster': gmm_cluster,
            'raw_hit': 1 if raw_err <= 0.01 else 0
        }
        
    overall_raw_hit = np.mean([info['raw_hit'] for info in trajectory_data.values()])
    print(f"\nStep 44 Baseline (Anisotropic Smoothed only, no blending) Overall Hit@1cm: {overall_raw_hit:.4%}")
    
    cluster_results = {}
    regime_names = {
        0: "fast_straight_low",
        1: "slow_moderate_turning",
        2: "fast_moderate_turning",
        3: "fast_straight_high",
        4: "fast_extreme_turning",
        5: "slow_extreme_turning"
    }
    
    opt_weights = {}
    optimized_overall_hits = []
    
    for cluster_id in range(6):
        c_name = regime_names[cluster_id]
        cluster_trajs = {fid: info for fid, info in trajectory_data.items() if info['gmm_cluster'] == cluster_id}
        
        if len(cluster_trajs) == 0:
            print(f"\n--- Cluster {cluster_id} ({c_name}): No trajectories found! ---")
            continue
            
        print(f"\n--- Optimizing Cluster {cluster_id} ({c_name}) | Trajectories: {len(cluster_trajs)} ---")
        raw_hit = np.mean([info['raw_hit'] for info in cluster_trajs.values()])
        print(f"  Raw minimum (no blending) Hit@1cm: {raw_hit:.4%}")
        
        best_hit, best_w = search_best_weights(cluster_trajs, true_targets, s4_priors, s7_priors, num_search_steps=20)
        print(f"  Optimized Hit@1cm: {best_hit:.4%} (weights: {best_w})")
        opt_weights[c_name] = best_w
        
        w_m, w_4, w_7 = best_w
        for fid, info in cluster_trajs.items():
            blended = w_m * info['coords'] + w_4 * s4_priors[fid] + w_7 * s7_priors[fid]
            err = np.linalg.norm(blended - true_targets[fid])
            optimized_overall_hits.append(1 if err <= 0.01 else 0)
            
    optimized_overall_hit_rate = np.mean(optimized_overall_hits)
    print(f"\n==================================================")
    print(f"FINAL OPTIMIZED OVERALL OOF Hit@1cm: {optimized_overall_hit_rate:.4%}")
    print(f"Baseline raw was: {overall_raw_hit:.4%}")
    print(f"Net validation improvement: {optimized_overall_hit_rate - overall_raw_hit:+.4%}")
    print(f"==================================================")
    
    print("\n--- Paste the following optimal weights dictionary in inference.py: ---")
    print("optimal_weights = {")
    for name, w in opt_weights.items():
        print(f"    \"{name}\": ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}),")
    print("}")

if __name__ == "__main__":
    main()
