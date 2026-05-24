import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker_v12():
    train_path = Path("step12/train_ranker.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step12/models/ranker_v4'
    
    print(f"Starting Step 12 Training (LPG + Dual Priors)...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='medium_quality',
        time_limit=1200, # 20 mins
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker_v12()
