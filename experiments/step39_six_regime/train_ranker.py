import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback
from pathlib import Path

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = None

REGIMES = [
    "fast_straight_low",
    "slow_moderate_turning",
    "fast_moderate_turning",
    "fast_straight_high",
    "fast_extreme_turning",
    "slow_extreme_turning"
]

def train_regime_model(regime_name, train_data_path, model_path, time_limit):
    try:
        send_discord_notification(URL, f"🚀 [Step 39] GMM 6-Regime Ranker ({regime_name}) Training Started...")
        print(f"\n==================================================")
        print(f"Loading {regime_name} data from {train_data_path}...")
        df = pd.read_csv(train_data_path)
        
        # Drop identifier and auxiliary regression target for training
        train_df = df.drop(columns=['id', 'reg_target'])
        
        # Memory-safe configuration (No RF/XT to prevent bad allocation on Windows)
        hyperparameters = {
            'GBM': {},
            'CAT': {},
            'XGB': {},
            'NN_TORCH': {},
            'FASTAI': {}
        }
        
        print(f"Starting AutoGluon TabularPredictor fit ('best_quality') for {regime_name}...")
        predictor = TabularPredictor(
            label='target',
            eval_metric='roc_auc',
            path=model_path
        ).fit(
            train_data=train_df,
            presets='best_quality',
            hyperparameters=hyperparameters,
            num_bag_folds=5,  # 5-fold bagging for memory safety
            time_limit=time_limit,
            ag_args_ensemble={
                'num_folds_parallel': 1,
                'fold_fitting_strategy': 'sequential_local'
            },
            verbosity=2
        )
        
        # Calculate OOF Hit@1cm validation score (Group-wise alignment)
        print("Calculating Out-Of-Fold (OOF) Hit@1cm metric...")
        oof_pred_proba = predictor.predict_proba_oof()
        score_col = 1 if 1 in oof_pred_proba.columns else oof_pred_proba.columns[0]
        df['oof_prob'] = oof_pred_proba[score_col].values
        
        # Select candidate with the highest OOF probability per trajectory
        idx_best = df.groupby('id')['oof_prob'].idxmax()
        best_cands = df.loc[idx_best]
        
        # Evaluate 1cm hit rate (reg_target is candidate distance in meters, threshold is 1cm = 0.01)
        oof_hit_rate = (best_cands['reg_target'] <= 0.01).mean()
        
        # Get AutoGluon Leaderboard
        leaderboard = predictor.leaderboard(train_df, silent=True)
        leaderboard_str = leaderboard.head(15).to_string()
        print(f"\n=== Model Leaderboard ({regime_name}) ===")
        print(leaderboard_str)
        
        best_model = predictor.model_best
        best_score = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
        
        success_msg = (
            f"✅ [Step 39] GMM 6-Regime Ranker ({regime_name}) Training Finished!\n"
            f"Group-wise **OOF Hit@1cm**: **{oof_hit_rate:.4%}** 🏆\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation AUC: **{best_score:.6f}**\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
        # Save OOF predictions to help verify overall OOF score later
        best_cands.to_csv(f"step39_six_regime/data/oof_best_{regime_name}.csv", index=False)
        
    except BaseException as e:
        error_msg = f"❌ [Step 39] Training ERROR ({regime_name}):\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

def main():
    # Make sure output models directory exists
    Path("step39_six_regime/models").mkdir(parents=True, exist_ok=True)
    
    # Train each regime model sequentially
    # Set time limit to 1500 seconds (25 minutes) per model to allow robust fitting within 2.5 hours total
    time_limit_per_model = 1500
    
    for regime in REGIMES:
        train_regime_model(
            regime_name=regime,
            train_data_path=f"step39_six_regime/data/train_ranker_v39_{regime}.csv",
            model_path=f"step39_six_regime/models/ranker_v39_{regime}",
            time_limit=time_limit_per_model
        )
        
    # Calculate overall OOF score by combining all OOF files
    print("\n=== Calculating Overall GMM 6-Regime OOF Score ===")
    oof_dfs = []
    for regime in REGIMES:
        fpath = Path(f"step39_six_regime/data/oof_best_{regime}.csv")
        if fpath.exists():
            oof_dfs.append(pd.read_csv(fpath))
            
    if len(oof_dfs) == len(REGIMES):
        combined = pd.concat(oof_dfs)
        overall_hit_rate = (combined['reg_target'] <= 0.01).mean()
        summary_msg = f"🏆 [Step 39] Overall 6-Regime GMM **Combined OOF Hit@1cm**: **{overall_hit_rate:.4%}**"
        send_discord_notification(URL, summary_msg)
        print(summary_msg)

if __name__ == "__main__":
    main()
