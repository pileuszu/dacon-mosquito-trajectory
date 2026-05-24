import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker():
    train_path = Path("step11/train_ranker.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step11/models/ranker_v3'
    
    print(f"Starting Stable Training (v3) on {len(train_data)} rows...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='medium_quality', # Stable on Windows, no bagging/stacking by default
        time_limit=600, # 10 mins is enough for 166k rows
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker()
