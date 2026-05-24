import pandas as pd
import os
import sys
import traceback
from pathlib import Path
from autogluon.tabular import TabularPredictor
from utils.notifier import send_discord_notification

sys.path.append(os.getcwd())
URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def train_hard_ranker_v19():
    try:
        send_discord_notification(URL, "🚀 [Step 19] Hard Target (0.5cm) Ranker Training Started...")
        
        train_path = Path("step19_hard_target/train_ranker_v19.csv")
        df = pd.read_csv(train_path)
        train_data = df.drop(columns=['id'])
        
        save_path = 'step19_hard_target/models/ranker_v19'
        
        # We use roc_auc as eval metric since data is highly imbalanced now (positives are rare)
        predictor = TabularPredictor(
            label='target', 
            eval_metric='roc_auc',
            path=save_path
        ).fit(
            train_data,
            presets='best_quality',
            time_limit=14400, # 4 Hours
            ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
        )
        
        send_discord_notification(URL, "✅ [Step 19] Hard Target Ranker Training Finished Successfully!")
        print(predictor.leaderboard())
        
    except Exception as e:
        error_msg = f"❌ [Step 19] Ranker Training ERROR:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        raise e

if __name__ == "__main__":
    train_hard_ranker_v19()
