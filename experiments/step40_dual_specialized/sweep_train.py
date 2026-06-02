import pandas as pd
import numpy as np
import sys
import os
import shutil
import traceback
from pathlib import Path
from autogluon.tabular import TabularPredictor

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

def run_sweep():
    # Make sure output models directory exists
    models_dir = Path("step40_dual_specialized/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    thresholds = [0.95, 0.98, 0.99]
    regimes = ["cruising", "gliding", "steering"]
    
    # We will log the results of each threshold
    sweep_results = []
    
    time_limit_per_model = 180  # 3 minutes per model for fast sweep
    hyperparameters = {
        'GBM': {},
        'CAT': {},
        'XGB': {}
    }
    
    send_discord_notification(None, "🚀 [Step 40 Sweep] Starting Denoising Threshold Sweep...")
    
    for th in thresholds:
        print(f"\n==================================================")
        print(f"=== Evaluating Denoising Threshold = {th} ===")
        print(f"==================================================")
        
        combined_dfs = []
        
        for regime in regimes:
            print(f"\n--- Training {regime} Model for Th={th} ---")
            train_path = f"step40_dual_specialized/data/train_ranker_v40_{regime}_th{th}.csv"
            val_path = f"step40_dual_specialized/data/train_ranker_v40_{regime}_th0.0.csv"  # Full validation set
            
            df_train = pd.read_csv(train_path)
            df_val = pd.read_csv(val_path)
            
            # 1. Train predictor on df_train
            # Drop identifier and auxiliary regression target for training, keep weight and target
            train_features = df_train.drop(columns=['id', 'reg_target'])
            
            temp_model_path = models_dir / f"temp_sweep_{th}_{regime}"
            if temp_model_path.exists():
                shutil.rmtree(temp_model_path)
                
            predictor = TabularPredictor(
                label='target',
                eval_metric='roc_auc',
                sample_weight='weight',
                path=str(temp_model_path)
            ).fit(
                train_data=train_features,
                presets='high_quality',
                hyperparameters=hyperparameters,
                num_bag_folds=5,
                time_limit=time_limit_per_model,
                ag_args_ensemble={
                    'num_folds_parallel': 1,
                    'fold_fitting_strategy': 'sequential_local'
                },
                verbosity=0
            )
            
            # 2. Get OOF prediction for trained samples
            oof_pred_proba = predictor.predict_proba_oof()
            score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[-1]
            df_train['pred_prob'] = oof_pred_proba[score_col].values
            
            # 3. Predict for excluded validation samples
            df_excl = df_val[~df_val['id'].isin(df_train['id'])].copy()
            if len(df_excl) > 0:
                excl_features = df_excl.drop(columns=['id', 'reg_target', 'target'])
                excl_pred_proba = predictor.predict_proba(excl_features)
                df_excl['pred_prob'] = excl_pred_proba[score_col].values
            
            # 4. Combine OOF and out-of-sample predictions
            df_merged = pd.concat([df_train, df_excl])
            
            # Save predictions for overall metric calculation
            combined_dfs.append(df_merged)
            
            # Clean up temp model directory to save space
            if temp_model_path.exists():
                shutil.rmtree(temp_model_path)
                
        # Calculate combined performance for this threshold
        combined_df = pd.concat(combined_dfs).reset_index(drop=True)
        idx_best = combined_df.groupby('id')['pred_prob'].idxmax()
        best_cands = combined_df.loc[idx_best]
        
        overall_hit = (best_cands['reg_target'] <= 0.01).mean()
        
        regime_breakdowns = []
        for name, gp in best_cands.groupby('regime'):
            r_hit = (gp['reg_target'] <= 0.01).mean()
            mean_err = gp['reg_target'].mean() * 100
            regime_breakdowns.append(f"{name}: {r_hit:.4%} (N={len(gp)})")
            
        summary_str = f"Threshold {th:.2f} -> Combined Hit@1cm: {overall_hit:.4%} ({', '.join(regime_breakdowns)})"
        print(f"\n📢 [Result] {summary_str}")
        send_discord_notification(None, f"📊 [Step 40 Sweep] {summary_str}")
        
        sweep_results.append({
            "threshold": th,
            "overall_hit": overall_hit,
            "regime_breakdowns": regime_breakdowns
        })
        
    # Print and notify final sweep summary
    print("\n==================================================")
    print("=== Denoising Threshold Sweep Summary ===")
    print("==================================================")
    best_th = None
    best_hit = -1
    summary_lines = []
    for res in sweep_results:
        line = f"Threshold {res['threshold']:.2f} -> Hit@1cm: {res['overall_hit']:.4%} ({', '.join(res['regime_breakdowns'])})"
        print(line)
        summary_lines.append(line)
        if res['overall_hit'] > best_hit:
            best_hit = res['overall_hit']
            best_th = res['threshold']
            
    final_msg = (
        f"🏆 [Step 40 Sweep Finished]\n"
        f"Best Threshold: **{best_th:.2f}** with Hit@1cm: **{best_hit:.4%}**\n\n"
        f"**Sweep Breakdown:**\n" + "\n".join(summary_lines)
    )
    send_discord_notification(None, final_msg)
    
if __name__ == "__main__":
    try:
        run_sweep()
    except BaseException as e:
        error_msg = f"❌ [Step 40 Sweep] ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(None, error_msg)
        print(error_msg)
        raise e
