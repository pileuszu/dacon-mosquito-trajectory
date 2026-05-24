import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
import sys
import os

sys.path.append(os.getcwd())
from step29_kinematic_scaling.physics import make_candidates, extract_multi_scale_derivatives, EPS

def evaluate_step29_oof():
    print("Loading Step 29 training data and predictions...")
    train_data_path = 'step29_kinematic_scaling/train_ranker_v29.csv'
    df = pd.read_csv(train_data_path)
    
    model_path = 'step29_kinematic_scaling/models/ranker_v29'
    predictor = TabularPredictor.load(model_path)
    
    # 1. Get OOF Probabilities
    print("Extracting OOF probabilities...")
    oof_pred_proba = predictor.predict_proba_oof()
    df['oof_prob'] = oof_pred_proba[1].values
    
    # 2. Raw Hit@1cm and L2 error
    print("Calculating raw OOF metrics...")
    idx_best_raw = df.groupby('id')['oof_prob'].idxmax()
    best_cands_raw = df.loc[idx_best_raw]
    
    raw_hit_rate = (best_cands_raw['reg_target'] <= 0.01).mean()
    raw_mean_l2 = best_cands_raw['reg_target'].mean() * 100 # in cm
    
    print(f"Raw OOF Hit@1cm: {raw_hit_rate:.4%}")
    print(f"Raw OOF Mean L2 Error: {raw_mean_l2:.4f} cm")
    
    # 3. Slow vs Fast raw Hit@1cm
    slow_mask = best_cands_raw['ctx_speed'] <= 0.0234
    slow_hit = (best_cands_raw.loc[slow_mask, 'reg_target'] <= 0.01).mean()
    fast_hit = (best_cands_raw.loc[~slow_mask, 'reg_target'] <= 0.01).mean()
    print(f"Raw OOF Hit@1cm (Slow, speed <= 2.34cm/s): {slow_hit:.4%}")
    print(f"Raw OOF Hit@1cm (Fast, speed > 2.34cm/s): {fast_hit:.4%}")
    
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
        
        # Soft state probability
        ctx = extract_multi_scale_derivatives(xyz)
        p_sacc = ctx["ctx_p_saccade"]
        
        # Dynamic Anisotropic SIGMA scaled by S_grid ** 1.5
        S_grid = 1.0 + 0.6 * p_sacc
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
        
        if speed <= 0.0234:
            slow_blended_hits.append(is_hit)
        else:
            fast_blended_hits.append(is_hit)
            
    blended_hit_rate = np.mean(blended_hits)
    blended_mean_l2 = np.mean(blended_l2_errors) * 100
    print(f"\nBlended OOF Hit@1cm: {blended_hit_rate:.4%}")
    print(f"Blended OOF Mean L2 Error: {blended_mean_l2:.4f} cm")
    print(f"Blended OOF Hit@1cm (Slow): {np.mean(slow_blended_hits):.4%}")
    print(f"Blended OOF Hit@1cm (Fast): {np.mean(fast_blended_hits):.4%}")

if __name__ == "__main__":
    evaluate_step29_oof()
