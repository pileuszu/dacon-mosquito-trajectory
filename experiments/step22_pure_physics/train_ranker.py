import os
import sys
sys.path.append(os.getcwd())

import pandas as pd
import traceback
from pathlib import Path
from autogluon.tabular import TabularPredictor
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_pure_physics_ranker_v22():
    try:
        send_discord_notification(URL, "🚀 [Step 22] Pure Physics Ranker V22 Training Started...")
        
        train_path = Path("step22_pure_physics/train_ranker_v22.csv")
        df = pd.read_csv(train_path)
        
        # Drop ID and reg_target (not used in feature training)
        train_data = df.drop(columns=['id', 'reg_target'])
        
        save_path = 'step22_pure_physics/models/ranker_v22'
        
        # Optimize ROC-AUC for ranking capability
        predictor = TabularPredictor(
            label='target', 
            eval_metric='roc_auc',
            path=save_path
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=10800, # 3 Hours max limit
            ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
        )
        
        leaderboard = predictor.leaderboard()
        leaderboard_str = leaderboard.to_string()
        
        success_msg = f"✅ [Step 22] Pure Physics Ranker V22 Training Finished Successfully!\n\n**Leaderboard Summary:**\n```\n{leaderboard_str[:1500]}\n```"
        send_discord_notification(URL, success_msg)
        print(predictor.leaderboard())
        
    except Exception as e:
        error_msg = f"❌ [Step 22] Ranker V22 Training ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        print(error_msg)
        raise e

if __name__ == "__main__":
    train_pure_physics_ranker_v22()
