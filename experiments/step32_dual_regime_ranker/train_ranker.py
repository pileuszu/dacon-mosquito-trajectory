import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_regime_model(regime_name, train_data_path, model_path, time_limit):
    try:
        send_discord_notification(URL, f"🚀 [Step 32] Regime Ranker V32 ({regime_name}) Training Started...")
        
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
        df['oof_prob'] = oof_pred_proba[1].values
        
        # Select candidate with the highest OOF probability per trajectory
        idx_best = df.groupby('id')['oof_prob'].idxmax()
        best_cands = df.loc[idx_best]
        
        # Evaluate 1cm hit rate
        oof_hit_rate = (best_cands['reg_target'] <= 0.01).mean()
        
        # Get AutoGluon Leaderboard
        leaderboard = predictor.leaderboard(train_df, silent=True)
        leaderboard_str = leaderboard.head(15).to_string()
        print(f"\n=== Model Leaderboard ({regime_name}) ===")
        print(leaderboard_str)
        
        best_model = predictor.model_best
        best_score = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
        
        success_msg = (
            f"✅ [Step 32] Regime Ranker V32 ({regime_name}) Training Finished!\n"
            f"Group-wise **OOF Hit@1cm**: **{oof_hit_rate:.4%}** 🏆\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation AUC: **{best_score:.6f}**\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(URL, success_msg)
        print(success_msg)
        
    except BaseException as e:
        error_msg = f"❌ [Step 32] Training ERROR ({regime_name}):\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

def main():
    # Sequential training of Slow and Fast models to prevent resource exhaustion
    train_regime_model(
        regime_name="SLOW",
        train_data_path="step32_dual_regime_ranker/train_ranker_v32_slow.csv",
        model_path="step32_dual_regime_ranker/models/ranker_v32_slow",
        time_limit=5400
    )
    
    train_regime_model(
        regime_name="FAST",
        train_data_path="step32_dual_regime_ranker/train_ranker_v32_fast.csv",
        model_path="step32_dual_regime_ranker/models/ranker_v32_fast",
        time_limit=5400
    )

if __name__ == "__main__":
    main()
