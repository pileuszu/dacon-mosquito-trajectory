import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_adaptive_grid_ranker_v25():
    try:
        send_discord_notification(URL, "🚀 [Step 25] Adaptive Grid Ranker V25 Training Started...")
        
        train_data_path = 'step25_adaptive_grid/train_ranker_v25.csv'
        print(f"Loading data from {train_data_path}...")
        df = pd.read_csv(train_data_path)
        
        # Drop identifiers and auxiliary targets for training
        train_df = df.drop(columns=['id', 'reg_target'])
        
        # Set save path
        model_path = 'step25_adaptive_grid/models/ranker_v25'
        
        # Memory-safe hyperparameter configuration excluding memory-heavy Scikit-learn RF/XT
        hyperparameters = {
            'GBM': {},
            'CAT': {},
            'XGB': {},
            'NN_TORCH': {},
            'FASTAI': {}
        }
        
        print("Starting AutoGluon TabularPredictor fit ('best_quality')...")
        predictor = TabularPredictor(
            label='target',
            eval_metric='roc_auc',
            path=model_path
        ).fit(
            train_data=train_df,
            presets='best_quality',
            hyperparameters=hyperparameters,
            num_bag_folds=5,  # 5-fold bagging for memory safety
            time_limit=10800,  # 3 Hours Limit
            ag_args_ensemble={'num_folds_parallel': 1},  # Force sequential bagging folds to bypass Ray
            verbosity=2
        )
        
        # 1. Calculate OOF Hit@1cm (Exact Leaderboard Metric Alignment)
        print("Calculating Out-Of-Fold (OOF) Hit@1cm metric...")
        oof_pred_proba = predictor.predict_proba_oof()
        
        # Align probabilities back to original dataframe
        df['oof_prob'] = oof_pred_proba[1].values
        
        # For each trajectory id, select the candidate with the highest OOF probability
        idx_best = df.groupby('id')['oof_prob'].idxmax()
        best_cands = df.loc[idx_best]
        
        # Calculate validation Hit@1cm
        oof_hit_rate = (best_cands['reg_target'] <= 0.01).mean()
        
        # 2. Generate and log the leaderboard
        leaderboard = predictor.leaderboard(train_df, silent=True)
        leaderboard_str = leaderboard.head(15).to_string()
        print("\n=== Model Leaderboard ===")
        print(leaderboard_str)
        
        best_model = predictor.model_best
        best_score = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
        
        success_msg = (
            f"✅ [Step 25] Adaptive Grid Ranker V25 Training Finished!\n"
            f"Group-wise **OOF Hit@1cm**: **{oof_hit_rate:.4%}** 🏆\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation AUC: **{best_score:.6f}**\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(URL, success_msg)
        
    except BaseException as e:
        error_msg = f"❌ [Step 25] Training ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e


if __name__ == "__main__":
    train_adaptive_grid_ranker_v25()
