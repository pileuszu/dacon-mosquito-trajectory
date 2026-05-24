import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import traceback
from pathlib import Path
from autogluon.tabular import TabularPredictor
from utils.notifier import send_discord_notification
URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_hybrid_ranker_v20():
    try:
        send_discord_notification(URL, "🚀 [Step 20] Hybrid Physics-Guided Ranker V20 Training Started...")
        
        train_path = Path("step20_hybrid_physics/train_ranker_v20.csv")
        df = pd.read_csv(train_path)
        
        # Drop non-feature columns
        train_data = df.drop(columns=['id'])
        
        save_path = 'step20_hybrid_physics/models/ranker_v20'
        
        # Optimize ROC-AUC for ranking capability
        predictor = TabularPredictor(
            label='target', 
            eval_metric='roc_auc',
            path=save_path
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=14400, # 4 Hours max limit
            ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
        )
        
        leaderboard = predictor.leaderboard()
        leaderboard_str = leaderboard.to_string()
        
        success_msg = f"✅ [Step 20] Hybrid Ranker V20 Training Finished Successfully!\n\n**Leaderboard Summary:**\n```\n{leaderboard_str[:1500]}\n```"
        send_discord_notification(URL, success_msg)
        print(predictor.leaderboard())
        
    except Exception as e:
        error_msg = f"❌ [Step 20] Ranker V20 Training ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    train_hybrid_ranker_v20()
