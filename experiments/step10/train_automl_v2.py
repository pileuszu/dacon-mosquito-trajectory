import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker():
    train_path = Path("step10/train_ranker.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id', 'reg_target'])
    
    label = 'target'
    save_path = 'step10/models/ranker_v2'
    
    print(f"Starting Robust AutoML Training on {len(train_data)} rows...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='high_quality', # Better stability than best_quality on Windows
        time_limit=1800, # 30 mins is plenty for 157k rows
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5},
        num_stack_levels=0, # Avoid nested bagging if it's causing Ray crashes
        num_bag_folds=5 # Basic bagging for stability
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker()
