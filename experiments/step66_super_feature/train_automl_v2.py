import os
import sys
import json
import shutil
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification

def main():
    send_discord_notification(None, "🚀 Started: [Step 67 train_automl_v2.py] Training 5-Fold SF-AutoML with Anisotropic Blending...")
    
    try:
        data_dir = Path("step66_super_feature/data")
        models_dir = Path("step66_super_feature/models")
        models_dir.mkdir(exist_ok=True)
        
        print("Loading datasets...")
        train_df = pd.read_parquet(data_dir / "train_tabular_v2.parquet")
        train_full_df = pd.read_parquet(data_dir / "train_tabular_full_v2.parquet")
        
        print(f"  Train shape: {train_df.shape}")
        print(f"  Train Full shape: {train_full_df.shape}")
        
        train_x = np.load(data_dir / "train_x.npy")
        train_y = np.load(data_dir / "train_y.npy")
        train_candidates_hybrid = np.load(data_dir / "train_candidates_hybrid.npy") # [10000, 43, 3]
        
        # Calculate Frenet unit vectors for Anisotropic Blending
        print("Calculating Frenet vectors for anisotropic spatial density voting...")
        v_tr = (train_x[:, 1:] - train_x[:, :-1]) / 0.04
        a_tr = (v_tr[:, 1:] - v_tr[:, :-1]) / 0.04
        v_last_tr = v_tr[:, -1]
        a_last_tr = a_tr[:, -1]
        
        speeds = np.linalg.norm(v_last_tr, axis=1)
        T_tr = v_last_tr / (speeds[:, None] + 1e-8)
        a_par = np.sum(a_last_tr * T_tr, axis=1)
        a_perp = a_last_tr - a_par[:, None] * T_tr
        a_perp_norm = np.linalg.norm(a_perp, axis=1)
        N_tr = a_perp / (a_perp_norm[:, None] + 1e-8)
        
        fallback = np.zeros_like(N_tr)
        axis = np.argmin(np.abs(T_tr), axis=1)
        fallback[np.arange(N_tr.shape[0]), axis] = 1.0
        fallback = fallback - np.sum(fallback * T_tr, axis=1)[:, None] * T_tr
        fallback = fallback / (np.linalg.norm(fallback, axis=1)[:, None] + 1e-8)
        N_tr = np.where(a_perp_norm[:, None] > 1e-6, N_tr, fallback)
        
        B_tr = np.cross(T_tr, N_tr, axis=1)
        B_tr = B_tr / (np.linalg.norm(B_tr, axis=1)[:, None] + 1e-8)
        
        folds = 5
        oof_preds_list = []
        
        print(f"\n=================== Starting Manual {folds}-Fold Training ===================")
        
        for fold in range(folds):
            print(f"\n--- Fold {fold+1} / {folds} ---")
            
            fold_model_path = models_dir / f"autogluon_fold_v2_{fold}"
            if fold_model_path.exists():
                print(f"Cleaning existing model directory: {fold_model_path}")
                shutil.rmtree(fold_model_path)
                
            # Split train / val
            train_fold_df = train_df[train_df['fold_id'] != fold]
            val_fold_full_df = train_full_df[train_full_df['fold_id'] == fold]
            
            print(f"  Train fold shape: {train_fold_df.shape}")
            print(f"  Validation fold shape (Full 43 candidates): {val_fold_full_df.shape}")
            
            train_data = train_fold_df.drop(columns=['sample_idx', 'cand_idx', 'fold_id'])
            
            predictor = TabularPredictor(
                label='dist_target',
                problem_type='regression',
                eval_metric='mean_absolute_error',
                path=fold_model_path
            )
            
            predictor.fit(
                train_data,
                time_limit=300,
                excluded_model_types=['RF', 'XT', 'NN_TORCH', 'FASTAI'],
                ag_args_fit={'use_ray': False},
                num_stack_levels=0,
                verbosity=2
            )
            
            # Predict validation fold (Full 43 candidates)
            val_data = val_fold_full_df.drop(columns=['sample_idx', 'cand_idx', 'fold_id', 'dist_target'])
            
            print(f"  Predicting OOF validation for Fold {fold+1}...")
            pred_dists = predictor.predict(val_data)
            
            val_eval = val_fold_full_df[['sample_idx', 'cand_idx', 'dist_target']].copy()
            val_eval['pred_dist'] = pred_dists
            oof_preds_list.append(val_eval)
            
        print("\n=================== Folds Training Complete ===================")
        
        oof_df = pd.concat(oof_preds_list, ignore_index=True)
        
        # ------------------ Apply Anisotropic Spatial Blending on OOF ------------------
        print("\nApplying Anisotropic Spatial Blending to resolve selection errors...")
        
        # Setup bandwidth hyperparameters (from V28 optimal settings)
        sigma_L = 0.007 # Longitudinal width (0.7 cm)
        sigma_T = 0.010 # Transverse width (1.0 cm)
        
        hits_argmax = 0
        hits_blended = 0
        num_samples = len(train_y)
        oof_coords_1step = np.zeros_like(train_y)
        
        for s_idx in range(num_samples):
            sample_preds = oof_df[oof_df['sample_idx'] == s_idx].sort_values('cand_idx')
            
            cands = train_candidates_hybrid[s_idx] # [43, 3]
            preds = sample_preds['pred_dist'].values # [43] in cm
            
            # 1. Argmax Baseline (Raw predictor Minimum distance)
            raw_best_idx = np.argmin(preds)
            raw_coord = cands[raw_best_idx]
            y_true = train_y[s_idx]
            if np.linalg.norm(raw_coord - y_true) <= 0.01:
                hits_argmax += 1
                
            # 2. Anisotropic Spatial Blending
            # Convert continuous predicted distances to weights
            w = np.exp(-preds / 1.5) # Soft temperature scaling
            
            # Compute cross-distances difference matrix [43, 43, 3]
            diffs = cands[:, None, :] - cands[None, :, :]
            
            T = T_tr[s_idx]
            N = N_tr[s_idx]
            B = B_tr[s_idx]
            
            # Project diffs to Frenet axes
            d_T = np.sum(diffs * T, axis=2)
            d_N = np.sum(diffs * N, axis=2)
            d_B = np.sum(diffs * B, axis=2)
            
            # Evaluate Anisotropic exponent
            exponent = -0.5 * ( (d_T**2 / (sigma_L**2)) + (d_N**2 / (sigma_T**2)) + (d_B**2 / (sigma_T**2)) )
            kernel_val = np.exp(exponent)
            
            # Sum weighted contributions
            scores = np.sum(w[None, :] * kernel_val, axis=1) # [43]
            
            # Select peak coordinates
            blended_best_idx = np.argmax(scores)
            blended_coord = cands[blended_best_idx]
            
            if np.linalg.norm(blended_coord - y_true) <= 0.01:
                hits_blended += 1
                
            oof_coords_1step[s_idx] = blended_coord
            
        hr_argmax = (hits_argmax / num_samples) * 100.0
        hr_blended = (hits_blended / num_samples) * 100.0
        
        print(f"\n=================== SF-AutoML STEP 67 OOF RESULTS ===================")
        print(f"Overall OOF Hit Rate@1cm (Argmax Baseline): {hr_argmax:.5f}% (Hits: {hits_argmax}/{num_samples})")
        print(f"Overall OOF Hit Rate@1cm (Anisotropic Blended): {hr_blended:.5f}% (Hits: {hits_blended}/{num_samples})")
        print(f"=====================================================================")
        
        np.save(data_dir / "oof_preds_sf_automl_v2.npy", oof_coords_1step)
        
        success_msg = f"✅ Finished: [Step 67] 5-Fold training complete. 참 OOF Hit Rate@1cm: {hr_blended:.3f}% (Argmax: {hr_argmax:.3f}%)"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 67]\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
