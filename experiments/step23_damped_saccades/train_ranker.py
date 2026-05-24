import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_pure_physics_ranker_v23():
    try:
        send_discord_notification(URL, "🚀 [Step 23] Damped Saccades Ranker V23 Training Started...")
        
        train_data_path = 'step23_damped_saccades/train_ranker_v23.csv'
        print(f"Loading data from {train_data_path}...")
        df = pd.read_csv(train_data_path)
        
        # Drop identifiers and auxiliary targets
        train_df = df.drop(columns=['id', 'reg_target'])
        
        # Set save path
        model_path = 'step23_damped_saccades/models/ranker_v23'
        
        print("Starting AutoGluon TabularPredictor fit ('best_quality')...")
        predictor = TabularPredictor(
            label='target',
            eval_metric='roc_auc',
            path=model_path
        ).fit(
            train_data=train_df,
            presets='best_quality',
            time_limit=10800,  # 3 Hours Limit
            verbosity=2
        )
        
        # Generate and log the leaderboard
        leaderboard = predictor.leaderboard(train_df, silent=True)
        leaderboard_str = leaderboard.head(15).to_string()
        print("\n=== Model Leaderboard ===")
        print(leaderboard_str)
        
        best_model = predictor.model_best
        best_score = leaderboard.loc[leaderboard['model'] == best_model, 'score_val'].values[0]
        
        success_msg = (
            f"✅ [Step 23] Damped Saccades Ranker V23 Training Finished!\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation AUC: **{best_score:.6f}**\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(URL, success_msg)
        
    except Exception as e:
        error_msg = f"❌ [Step 23] Training ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    train_pure_physics_ranker_v23()
