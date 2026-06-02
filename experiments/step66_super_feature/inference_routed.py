import os
import sys
import json
import csv
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification

def main():
    send_discord_notification(None, "🚀 Started: [Step 67 Routed Inference] Generating submission using Hybrid Routed Strategy...")
    
    try:
        data_dir = Path("step66_super_feature/data")
        models_dir = Path("step66_super_feature/models")
        out_dir = Path("outputs/step66_super_feature")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load test candidates reference [10000, 43, 3]
        print("Loading hybrid candidates reference...")
        test_candidates_hybrid = np.load(data_dir / "test_candidates_hybrid_v3.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        test_x = np.load(data_dir / "test_x.npy")
        test_cfm_preds = np.load(data_dir / "test_cfm_preds.npy") # CFM 2step [10000, 3] from step65
        
        # 1. Classify Easy/Hard based on Candidate Spread
        cand_std = np.std(test_candidates_hybrid, axis=1) # [10000, 3]
        cand_spread = np.mean(cand_std, axis=1) # [10000]
        
        # We use 0.38cm threshold (derived from 75% quantile of train set)
        easy_thresh = 0.0038 # 0.38cm in meters
        easy_mask = cand_spread <= easy_thresh
        num_easy = np.sum(easy_mask)
        print(f"Test Set Split: Easy={num_easy} ({num_easy/len(test_ids)*100.0:.1f}%), Hard={len(test_ids)-num_easy} ({(len(test_ids)-num_easy)/len(test_ids)*100.0:.1f}%)")
        
        # 2. Get CFM-Nearest Candidate predictions for Easy cases
        print("Computing CFM-Nearest Candidate coordinates...")
        cfm_nearest_coords = np.zeros((len(test_ids), 3))
        for idx in range(len(test_ids)):
            cands = test_candidates_hybrid[idx]
            cfm_pred = test_cfm_preds[idx]
            dists = np.linalg.norm(cands - cfm_pred, axis=1)
            cfm_nearest_coords[idx] = cands[np.argmin(dists)]
            
        # 3. Predict with AutoML & Spatial Blending for Hard cases
        print("Predicting with AutoML folds for Hard cases...")
        test_df = pd.read_parquet(data_dir / "test_tabular_v3.parquet")
        test_data = test_df.drop(columns=['sample_idx', 'cand_idx'])
        
        folds = 5
        pred_dists_list = []
        for fold in range(folds):
            fold_model_path = models_dir / f"autogluon_fold_v3_{fold}"
            predictor = TabularPredictor.load(fold_model_path)
            fold_preds = predictor.predict(test_data)
            pred_dists_list.append(fold_preds)
            
        avg_pred_dists = np.mean(pred_dists_list, axis=0)
        test_df['pred_dist'] = avg_pred_dists
        
        # Calculate Frenet vectors for Anisotropic Blending
        print("Calculating Frenet vectors for test set...")
        v_te = (test_x[:, 1:] - test_x[:, :-1]) / 0.04
        a_te = (v_te[:, 1:] - v_te[:, :-1]) / 0.04
        v_last_te = v_te[:, -1]
        a_last_te = a_te[:, -1]
        
        speeds = np.linalg.norm(v_last_te, axis=1)
        T_te = v_last_te / (speeds[:, None] + 1e-8)
        a_par = np.sum(a_last_te * T_te, axis=1)
        a_perp = a_last_te - a_par[:, None] * T_te
        a_perp_norm = np.linalg.norm(a_perp, axis=1)
        N_te = a_perp / (a_perp_norm[:, None] + 1e-8)
        
        fallback = np.zeros_like(N_te)
        axis = np.argmin(np.abs(T_te), axis=1)
        fallback[np.arange(N_te.shape[0]), axis] = 1.0
        fallback = fallback - np.sum(fallback * T_te, axis=1)[:, None] * T_te
        fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + 1e-8)
        N_te = np.where(a_perp_norm[:, None] > 1e-6, N_te, fallback)
        
        B_te = np.cross(T_te, N_te, axis=1)
        B_te = B_te / (np.linalg.norm(B_te, axis=1)[:, None] + 1e-8)
        
        sigma_L = 0.015
        sigma_T = 0.025
        temp = 0.3
        
        automl_blended_coords = np.zeros((len(test_ids), 3))
        print("Applying Anisotropic Spatial Blending for Hard cases...")
        for s_idx in range(len(test_ids)):
            if not easy_mask[s_idx]: # Only blend for hard cases to save time, but can do all
                sample_preds = test_df[test_df['sample_idx'] == s_idx].sort_values('cand_idx')
                cands = test_candidates_hybrid[s_idx]
                preds = sample_preds['pred_dist'].values
                
                w = np.exp(-preds / temp)
                diffs = cands[:, None, :] - cands[None, :, :]
                
                T = T_te[s_idx]
                N = N_te[s_idx]
                B = B_te[s_idx]
                
                d_T = np.sum(diffs * T, axis=2)
                d_N = np.sum(diffs * N, axis=2)
                d_B = np.sum(diffs * B, axis=2)
                
                exponent = -0.5 * ( (d_T**2 / (sigma_L**2)) + (d_N**2 / (sigma_T**2)) + (d_B**2 / (sigma_T**2)) )
                kernel_val = np.exp(exponent)
                
                scores = np.sum(w[None, :] * kernel_val, axis=1)
                
                blended_best_idx = np.argmax(scores)
                automl_blended_coords[s_idx] = cands[blended_best_idx]
                
        # 4. Route: Easy -> CFM-Nearest, Hard -> AutoML Blended
        submission_coords = np.zeros((len(test_ids), 3))
        for idx in range(len(test_ids)):
            if easy_mask[idx]:
                submission_coords[idx] = cfm_nearest_coords[idx]
            else:
                submission_coords[idx] = automl_blended_coords[idx]
                
        # Build final submission CSV file
        sub_path = out_dir / "submission_routed_hybrid.csv"
        print(f"Writing routed submission to {sub_path}...")
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, coord in zip(test_ids, submission_coords):
                writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
                
        print("\n=================== Routed Inference Summary ===================")
        print(f"Routed submission generated successfully at: {sub_path}")
        
        df_sub = pd.read_csv(sub_path)
        print(f"  Shape: {df_sub.shape}")
        print(f"  NaN count: {df_sub.isna().sum().to_dict()}")
        p_last_te_arr = np.load(data_dir / "test_x.npy")[:, -1]
        diffs = np.linalg.norm(submission_coords - p_last_te_arr, axis=1)
        disp_stats = f"Displacement stats (cm) -> Mean: {diffs.mean()*100:.3f}, Max: {diffs.max()*100:.3f}, Std: {diffs.std()*100:.3f}"
        print(f"  {disp_stats}")
        print("=========================================================")
        
        success_msg = f"✅ Finished: [Step 67 Routed] Submission generated. Mean displacement: {diffs.mean()*100:.3f}cm"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 67 Routed] Routed Inference failed.\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
