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
    send_discord_notification(None, "🚀 Started: [Step 67 inference_v2.py] Generating final submission coordinates with 5-Fold soft ensemble & Anisotropic Blending...")
    
    try:
        data_dir = Path("step66_super_feature/data")
        models_dir = Path("step66_super_feature/models")
        out_dir = Path("outputs/step66_super_feature")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print("Loading test tabular dataset...")
        test_df = pd.read_parquet(data_dir / "test_tabular_v2.parquet")
        print(f"  Loaded dataset shape: {test_df.shape}")
        
        test_data = test_df.drop(columns=['sample_idx', 'cand_idx'])
        
        folds = 5
        pred_dists_list = []
        
        # Predict using all 5 fold models
        print("Predicting distance error using 5-Fold models...")
        for fold in range(folds):
            fold_model_path = models_dir / f"autogluon_fold_v2_{fold}"
            print(f"  Loading predictor for Fold {fold+1} from {fold_model_path}...")
            predictor = TabularPredictor.load(fold_model_path)
            
            print(f"  Predicting with Fold {fold+1} model...")
            fold_preds = predictor.predict(test_data)
            pred_dists_list.append(fold_preds)
            
        # Average the predicted distances across folds
        print("Averaging predicted distances...")
        avg_pred_dists = np.mean(pred_dists_list, axis=0)
        test_df['pred_dist'] = avg_pred_dists
        
        # Load hybrid test candidates reference [10000, 43, 3]
        print("Loading hybrid candidates reference...")
        test_candidates_hybrid = np.load(data_dir / "test_candidates_hybrid.npy")
        with open(data_dir / "test_ids.json", "r") as f:
            test_ids = json.load(f)
            
        # Load test coordinates for Frenet vectors
        test_x = np.load(data_dir / "test_x.npy")
        
        # Calculate Frenet vectors for anisotropic spatial density voting
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
        
        # ------------------ Apply Anisotropic Spatial Blending on Test ------------------
        print("Resolving selection errors with Anisotropic Spatial Blending...")
        sigma_L = 0.007 # Longitudinal width (0.7 cm)
        sigma_T = 0.010 # Transverse width (1.0 cm)
        
        submission_coords = np.zeros((len(test_ids), 3))
        
        for s_idx in range(len(test_ids)):
            sample_preds = test_df[test_df['sample_idx'] == s_idx].sort_values('cand_idx')
            
            cands = test_candidates_hybrid[s_idx] # [43, 3]
            preds = sample_preds['pred_dist'].values # [43] in cm
            
            # soft temperature mapping
            w = np.exp(-preds / 1.5)
            
            # Cross distance differences
            diffs = cands[:, None, :] - cands[None, :, :]
            
            T = T_te[s_idx]
            N = N_te[s_idx]
            B = B_te[s_idx]
            
            # Project diffs
            d_T = np.sum(diffs * T, axis=2)
            d_N = np.sum(diffs * N, axis=2)
            d_B = np.sum(diffs * B, axis=2)
            
            # Spatial Gaussian voting
            exponent = -0.5 * ( (d_T**2 / (sigma_L**2)) + (d_N**2 / (sigma_T**2)) + (d_B**2 / (sigma_T**2)) )
            kernel_val = np.exp(exponent)
            
            scores = np.sum(w[None, :] * kernel_val, axis=1) # [43]
            
            blended_best_idx = np.argmax(scores)
            submission_coords[s_idx] = cands[blended_best_idx]
            
        # Build final submission CSV file
        sub_path = out_dir / "submission.csv"
        print(f"Writing final submission to {sub_path}...")
        with sub_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "x", "y", "z"])
            for sample_id, coord in zip(test_ids, submission_coords):
                writer.writerow([sample_id, f"{coord[0]:.9f}", f"{coord[1]:.9f}", f"{coord[2]:.9f}"])
                
        print("\n=================== Inference Summary ===================")
        print(f"Submission generated successfully at: {sub_path}")
        
        # Basic sanity checks on the generated submission
        df_sub = pd.read_csv(sub_path)
        print(f"  Shape: {df_sub.shape}")
        print(f"  NaN count: {df_sub.isna().sum().to_dict()}")
        p_last_te = np.load(data_dir / "test_x.npy")[:, -1]
        diffs = np.linalg.norm(submission_coords - p_last_te, axis=1)
        disp_stats = f"Displacement stats (cm) -> Mean: {diffs.mean()*100:.3f}, Max: {diffs.max()*100:.3f}, Std: {diffs.std()*100:.3f}"
        print(f"  {disp_stats}")
        print("=========================================================")
        
        success_msg = f"✅ Finished: [Step 67] Inference completed successfully. Mean displacement: {diffs.mean()*100:.3f}cm, Max: {diffs.max()*100:.3f}cm"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 67] Inference failed.\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
