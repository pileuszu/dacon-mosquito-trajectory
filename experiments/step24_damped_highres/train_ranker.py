import pandas as pd
from autogluon.tabular import TabularPredictor
import sys
import os
import traceback

sys.path.append(os.getcwd())
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_damped_highres_ranker_v24():
    try:
        send_discord_notification(URL, "🚀 [Step 24] Damped High-Res Ranker V24 Training Started...")
        
        train_data_path = 'step24_damped_highres/train_ranker_v24.csv'
        print(f"Loading data from {train_data_path}...")
        df = pd.read_csv(train_data_path)
        
        # Drop identifiers and auxiliary targets
        train_df = df.drop(columns=['id', 'reg_target'])
        
        # Set save path
        model_path = 'step24_damped_highres/models/ranker_v24'
        
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
            f"✅ [Step 24] Damped High-Res Ranker V24 Training Finished!\n"
            f"Best Model: **{best_model}**\n"
            f"Best Validation AUC: **{best_score:.6f}**\n\n"
            f"**Leaderboard Top 10:**\n```\n{leaderboard.head(10)[['model', 'score_val', 'pred_time_val', 'fit_time']].to_string()}\n```"
        )
        send_discord_notification(URL, success_msg)
        
    except Exception as e:
        error_msg = f"❌ [Step 24] Training ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    train_damped_highres_ranker_v24()
