import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import sys
import os

sys.path.append(os.getcwd())
from step33_curvature_adaptive.physics import make_candidates, extract_multi_scale_derivatives, EPS

def evaluate_step33_oof():
    print("Loading Step 33 training data and predictions...")
    df_slow = pd.read_csv('step33_curvature_adaptive/train_ranker_v33_slow.csv')
    df_fast = pd.read_csv('step33_curvature_adaptive/train_ranker_v33_fast.csv')
    
    print("Loading AutoGluon Predictors...")
    predictor_slow = TabularPredictor.load('step33_curvature_adaptive/models/ranker_v33_slow')
    predictor_fast = TabularPredictor.load('step33_curvature_adaptive/models/ranker_v33_fast')
    
    print("Extracting OOF probabilities...")
    oof_slow = predictor_slow.predict_proba_oof()
    df_slow['oof_prob'] = oof_slow[1].values
    
    oof_fast = predictor_fast.predict_proba_oof()
    df_fast['oof_prob'] = oof_fast[1].values
    
    df = pd.concat([df_slow, df_fast], ignore_index=True)
    
    # 2. Raw Hit@1cm and L2 error
    print("Calculating raw OOF metrics...")
    idx_best_raw = df.groupby('id')['oof_prob'].idxmax()
    best_cands_raw = df.loc[idx_best_raw]
    
    raw_hit_rate = (best_cands_raw['reg_target'] <= 0.01).mean()
    raw_mean_l2 = best_cands_raw['reg_target'].mean() * 100 # in cm
    
    print(f"Raw OOF Hit@1cm: {raw_hit_rate:.4%}")
    print(f"Raw OOF Mean L2 Error: {raw_mean_l2:.4f} cm")
    
    # 3. Slow vs Fast raw Hit@1cm (by speed)
    slow_mask = best_cands_raw['ctx_speed'] <= 0.0234
    slow_hit = (best_cands_raw.loc[slow_mask, 'reg_target'] <= 0.01).mean()
    fast_hit = (best_cands_raw.loc[~slow_mask, 'reg_target'] <= 0.01).mean()
    print(f"Raw OOF Hit@1cm (Slow speed <= 2.34cm/s): {slow_hit:.4%}")
    print(f"Raw OOF Hit@1cm (Fast speed > 2.34cm/s): {fast_hit:.4%}")
    
    # 4. Apply Anisotropic Blending to OOF Predictions
    print("Applying Anisotropic Blending to OOF predictions...")
    data_dir = Path("data/open")
    train_dir = data_dir / "train"
    labels_df = pd.read_csv(data_dir / "train_labels.csv")
    s4_preds_df = pd.read_csv("step12/step4_preds_train.csv").set_index('id')
    
    unique_ids = df['id'].unique()
    np.random.seed(42)
    unique_ids = np.random.choice(unique_ids, min(1000, len(unique_ids)), replace=False)
    
    blended_hits = []
    blended_l2_errors = []
    
    slow_blended_hits = []
    fast_blended_hits = []
    
    cruising_blended_hits = []
    turning_blended_hits = []
    
    for fid in tqdm(unique_ids):
        fpath = train_dir / f"{fid}.csv"
        traj_df = pd.read_csv(fpath)
        xyz = traj_df[['x', 'y', 'z']].to_numpy(dtype=np.float32)
        target = labels_df.loc[labels_df['id'] == fid, ['x', 'y', 'z']].to_numpy(dtype=np.float32)[0]
        
        # Reconstruct all candidates for this ID
        p0 = xyz[-1]
        last_vel = xyz[-1] - xyz[-2]
        s7_pos = p0 + 2.0 * last_vel
        s4_pos = s4_preds_df.loc[fid].to_numpy()
        priors = [s7_pos, s4_pos]
        
        cands, specs = make_candidates(xyz, priors=priors, end_idx=-1, horizon=2)
        
        # Get OOF probabilities for this ID
        sub_df = df[df['id'] == fid]
        probs = np.zeros(len(cands), dtype=np.float32)
        for _, r in sub_df.iterrows():
            idx_val = int(r['cand_idx'])
            if idx_val < len(probs):
                probs[idx_val] = r['oof_prob']
            
        # Physical values
        speed = np.linalg.norm(last_vel)
        speed_m_s = speed / 0.01
        tangent = last_vel / (speed + EPS)
        
        # Turn metrics
        ctx = extract_multi_scale_derivatives(xyz)
        p_sacc = ctx["ctx_p_saccade"]
        ctx_lat_accel = ctx["ctx_lat_accel"]
        
        # Terminal raw curvature
        acc = last_vel - (xyz[-2] - xyz[-3])
        cross_va = np.cross(last_vel, acc)
        ctx_curv = np.linalg.norm(cross_va) / (speed**3 + 1e-6)
        
        is_turning = (ctx_curv > 12.0) or (ctx_lat_accel > 0.0020)
        
        # Dynamic Anisotropic SIGMA scaled by S_grid ** 1.5
        if speed <= 0.0234 and not is_turning:
            S_grid = 1.0
        else:
            S_grid = float(np.clip(1.0 + 0.6 * p_sacc + 0.05 * (ctx_curv - 6.0), 1.0, 2.5))
            
        S_scale = S_grid ** 1.5
        sigma_tangential = np.clip(0.0035 + 0.005 * speed_m_s * p_sacc, 0.003, 0.011) * S_scale
        sigma_normal = np.clip(0.0035 + 0.0015 * speed_m_s * p_sacc, 0.003, 0.006) * S_scale
        
        # Anisotropic Blending
        cands_diff = cands[:, None, :] - cands[None, :, :]
        dx_tangential = np.dot(cands_diff, tangent)
        dx_sq = np.sum(cands_diff ** 2, axis=-1)
        dx_normal_sq = np.maximum(dx_sq - dx_tangential ** 2, 0.0)
        
        weights = np.exp(- (dx_tangential ** 2) / (2.0 * (sigma_tangential ** 2)) 
                          - dx_normal_sq / (2.0 * (sigma_normal ** 2)))
        
        smoothed_probs = weights.dot(probs)
        best_idx = np.argmax(smoothed_probs)
        final_coords = cands[best_idx]
        
        dist_to_target = np.linalg.norm(final_coords - target)
        is_hit = 1 if dist_to_target <= 0.01 else 0
        
        blended_hits.append(is_hit)
        blended_l2_errors.append(dist_to_target)
        
        # Speed-based groups (for direct comparison to step32/step30)
        if speed <= 0.0234:
            slow_blended_hits.append(is_hit)
        else:
            fast_blended_hits.append(is_hit)
            
        # Behavior-based groups
        if speed <= 0.0234 and not is_turning:
            cruising_blended_hits.append(is_hit)
        else:
            turning_blended_hits.append(is_hit)
            
    blended_hit_rate = np.mean(blended_hits)
    blended_mean_l2 = np.mean(blended_l2_errors) * 100
    print(f"\nBlended OOF Hit@1cm (Overall): {blended_hit_rate:.4%}")
    print(f"Blended OOF Mean L2 Error: {blended_mean_l2:.4f} cm")
    print(f"Blended OOF Hit@1cm (Speed Slow <= 2.34cm/s): {np.mean(slow_blended_hits):.4%}")
    print(f"Blended OOF Hit@1cm (Speed Fast > 2.34cm/s): {np.mean(fast_blended_hits):.4%}")
    print(f"Blended OOF Hit@1cm (Behavior Cruising): {np.mean(cruising_blended_hits):.4%}")
    print(f"Blended OOF Hit@1cm (Behavior Turning): {np.mean(turning_blended_hits):.4%}")

if __name__ == "__main__":
    evaluate_step33_oof()
