import os
import sys
import pickle
import traceback
import shutil
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

# Ensure working directory in path
sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

HYPERPARAMETERS = {
    'GBM': {},
    'CAT': {},
    'XGB': {}
}

MODELS_DIR = Path("step42_3d_out_of_plane/models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def add_fold_id_column(df, num_folds=5):
    unique_ids = sorted(df['id'].unique())
    id_to_fold = {}
    for uid in unique_ids:
        h = hashlib.md5(str(uid).encode('utf-8')).hexdigest()
        fold_idx = int(h, 16) % num_folds
        id_to_fold[uid] = fold_idx
    df['fold_id'] = df['id'].map(id_to_fold)
    return df

def apply_feature_engineering(df):
    df = df.copy()
    # Engineered speed/accel ratio features
    df['norm_cand_speed'] = df['cand_speed'] / (df['ctx_speed'] + 1e-6)
    df['norm_cand_accel'] = df['cand_accel'] / (df['ctx_acc'] + 1e-6)
    df['s7_s4_dist'] = np.abs(df['dist_to_s7'] - df['dist_to_s4'])
    
    # 3D out-of-plane specific engineered features
    df['norm_cand_disp_binormal'] = df['cand_disp_binormal'] / (df['ctx_speed'] + 1e-6)
    df['norm_cand_acc_binormal'] = df['cand_acc_binormal'] / (df['ctx_acc'] + 1e-6)
    return df

def evaluate_predictions(combined_df):
    idx_best = combined_df.groupby('id')['pred_prob'].idxmax()
    best_cands = combined_df.loc[idx_best]
    overall_hit = (best_cands['reg_target'] <= 0.01).mean()
    
    regime_breakdown = {}
    for name, gp in best_cands.groupby('regime'):
        r_hit = (gp['reg_target'] <= 0.01).mean()
        regime_breakdown[name] = (r_hit, len(gp))
        
    return overall_hit, best_cands, regime_breakdown

def main():
    try:
        msg = "🚀 Started: [Step 42 train_automl.py] Training 3D Out-of-Plane specialized rankers..."
        send_discord_notification(None, msg)
        print(msg)
        
        regimes = ["cruising", "gliding", "steering"]
        
        # 1. Train the specialized models on denoised datasets (Th=0.98)
        for regime in regimes:
            train_path = f"step42_3d_out_of_plane/data/train_ranker_v42_{regime}_th0.98.csv"
            df_train = pd.read_csv(train_path)
            
            # Apply feature engineering
            df_train = apply_feature_engineering(df_train)
            df_train = add_fold_id_column(df_train, num_folds=5)
            
            train_df = df_train.drop(columns=['reg_target', 'id'])
            
            model_path = MODELS_DIR / f"final_{regime}"
            if model_path.exists():
                try:
                    shutil.rmtree(model_path, ignore_errors=True)
                except Exception:
                    pass
                import time
                time.sleep(1)
                
            print(f"\n--- Training final model for {regime} with best_quality and groups='fold_id' ---")
            predictor = TabularPredictor(
                label='target',
                eval_metric='log_loss',
                sample_weight='weight',
                groups='fold_id',
                path=str(model_path)
            ).fit(
                train_data=train_df,
                presets='best_quality',
                hyperparameters=HYPERPARAMETERS,
                num_bag_folds=5,
                time_limit=1800,  # 30 minutes budget per model
                num_gpus=0,  # CPU only
                dynamic_stacking=False,
                ag_args_ensemble={
                    'num_folds_parallel': 1,
                    'fold_fitting_strategy': 'sequential_local'
                },
                verbosity=2
            )
            
        # 2. Evaluate unbiased OOF validation predictions (using Th=0.0 dataset)
        print("\n--- Generating validation predictions to assess unbiased OOF score ---")
        combined_dfs_final = []
        
        for regime in regimes:
            train_path = f"step42_3d_out_of_plane/data/train_ranker_v42_{regime}_th0.98.csv"
            val_path = f"step42_3d_out_of_plane/data/train_ranker_v42_{regime}_th0.0.csv"
            
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            
            df_train = apply_feature_engineering(df_train)
            df_val = apply_feature_engineering(df_val)
            
            df_train = add_fold_id_column(df_train, num_folds=5)
            
            model_path = MODELS_DIR / f"final_{regime}"
            predictor = TabularPredictor.load(str(model_path))
            
            # Predict OOF for train samples
            oof_pred_proba = predictor.predict_proba_oof()
            score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
            df_train['pred_prob'] = oof_pred_proba[score_col].values
            
            # Predict for out-of-sample validation trajectories (ambiguous trajectories excluded from train)
            df_excl = df_val[~df_val['id'].isin(df_train['id'])].copy()
            if len(df_excl) > 0:
                excl_features = df_excl.drop(columns=['reg_target', 'target', 'id', 'fold_id'], errors='ignore')
                excl_pred_proba = predictor.predict_proba(excl_features)
                df_excl['pred_prob'] = excl_pred_proba[score_col].values
                
            df_merged = pd.concat([df_train, df_excl])
            combined_dfs_final.append(df_merged)
            
        combined_df_final = pd.concat(combined_dfs_final).reset_index(drop=True)
        
        # Save predictions for blending weight optimization
        combined_df_final.to_csv("step42_3d_out_of_plane/data/oof_predictions_final.csv", index=False)
        print("\nSaved OOF predictions to step42_3d_out_of_plane/data/oof_predictions_final.csv")
        
        overall_hit, best_cands, breakdowns = evaluate_predictions(combined_df_final)
        
        success_msg = (
            f"✅ Finished: [Step 42 train_automl.py] Success!\n"
            f"Overall 3D Leak-Free OOF Hit@1cm: **{overall_hit:.4%}**\n"
            f"Regime Breakdown:\n"
        )
        for name, (r_hit, N) in breakdowns.items():
            success_msg += f"  * **{name}**: **{r_hit:.4%}** (N={N})\n"
            
        send_discord_notification(None, success_msg)
        print(success_msg)
        
    except BaseException as e:
        error_msg = f"❌ Failed: [Step 42 train_automl.py] ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    main()
