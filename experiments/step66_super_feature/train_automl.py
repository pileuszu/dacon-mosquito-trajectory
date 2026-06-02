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
    send_discord_notification(None, "🚀 Started: [Step 66 train_automl.py] Training Leakage-Free Manual 5-Fold SF-AutoML Ranker...")
    
    try:
        data_dir = Path("step66_super_feature/data")
        models_dir = Path("step66_super_feature/models")
        models_dir.mkdir(exist_ok=True)
        
        print("Loading datasets...")
        # Train dataset (Down-sampled, 8 candidates per sample)
        train_df = pd.read_parquet(data_dir / "train_tabular.parquet")
        # Train Full dataset (All 36 candidates per sample) for True OOF evaluation
        train_full_df = pd.read_parquet(data_dir / "train_tabular_full.parquet")
        
        print(f"  Train shape: {train_df.shape}")
        print(f"  Train Full shape: {train_full_df.shape}")
        
        train_y = np.load(data_dir / "train_y.npy")
        train_candidates = np.load(data_dir / "train_candidates.npy")
        
        folds = 5
        oof_preds_list = []
        
        print(f"\n=================== Starting Manual {folds}-Fold Training ===================")
        
        for fold in range(folds):
            print(f"\n--- Fold {fold+1} / {folds} ---")
            
            fold_model_path = models_dir / f"autogluon_fold_{fold}"
            if fold_model_path.exists():
                print(f"Cleaning existing model directory: {fold_model_path}")
                shutil.rmtree(fold_model_path)
                
            # Split train / val
            train_fold_df = train_df[train_df['fold_id'] != fold]
            # Validation set must contain all 36 candidates per sample to prevent leakage
            val_fold_full_df = train_full_df[train_full_df['fold_id'] == fold]
            
            print(f"  Train fold shape: {train_fold_df.shape}")
            print(f"  Validation fold shape (Full 36 candidates): {val_fold_full_df.shape}")
            
            train_data = train_fold_df.drop(columns=['sample_idx', 'cand_idx', 'fold_id'])
            
            predictor = TabularPredictor(
                label='dist_target',
                problem_type='regression',
                eval_metric='mean_absolute_error',
                path=fold_model_path
            )
            
            # Train the predictor for the current fold (fast training, bagging/stacking disabled)
            predictor.fit(
                train_data,
                time_limit=300, # 5 min limit per fold
                excluded_model_types=['RF', 'XT', 'NN_TORCH', 'FASTAI'],
                ag_args_fit={'use_ray': False},
                num_stack_levels=0,
                verbosity=2
            )
            
            # Predict for validation fold (Full 36 candidates)
            val_data = val_fold_full_df.drop(columns=['sample_idx', 'cand_idx', 'fold_id', 'dist_target'])
            
            print(f"  Predicting OOF validation for Fold {fold+1}...")
            pred_dists = predictor.predict(val_data)
            
            # Record prediction
            val_eval = val_fold_full_df[['sample_idx', 'cand_idx', 'dist_target']].copy()
            val_eval['pred_dist'] = pred_dists
            oof_preds_list.append(val_eval)
            
        print("\n=================== Folds Training Complete ===================")
        
        # Combine all OOF predictions (Full 36 candidates per sample for all 10,000 samples)
        oof_df = pd.concat(oof_preds_list, ignore_index=True)
        print(f"Overall OOF DataFrame shape: {oof_df.shape} (Expected: 360000 rows)")
        
        # Select the candidate with minimum predicted distance error per sample (36-candidate selection)
        best_cand_indices = oof_df.groupby('sample_idx')['pred_dist'].idxmin()
        best_rows = oof_df.loc[best_cand_indices]
        
        hits = 0
        num_samples = len(best_rows)
        oof_coords_1step = np.zeros_like(train_y)
        
        for _, row in best_rows.iterrows():
            s_idx = int(row['sample_idx'])
            c_idx = int(row['cand_idx'])
            
            selected_coord = train_candidates[s_idx, c_idx]
            y_true = train_y[s_idx]
            
            error = np.linalg.norm(selected_coord - y_true)
            if error <= 0.01:
                hits += 1
            oof_coords_1step[s_idx] = selected_coord
            
        overall_hr = (hits / num_samples) * 100.0
        
        print(f"\n=================== SF-AutoML TRUE OOF RESULTS ===================")
        print(f"Overall True OOF Hit Rate@1cm (36-candidate selection): {overall_hr:.5f}% (Hits: {hits}/{num_samples})")
        print(f"==================================================================")
        
        np.save(data_dir / "oof_preds_sf_automl.npy", oof_coords_1step)
        
        success_msg = f"✅ Finished: [Step 66 Revised] SF-AutoML manual 5-Fold training complete. Leakage-Free OOF Hit Rate@1cm: {overall_hr:.3f}%"
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 66 Revised]\nError: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
