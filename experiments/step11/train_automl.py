import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker():
    train_path = Path("step11/train_ranker.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step11/models/ranker'
    
    print(f"Starting Robust Stacking Training on {len(train_data)} rows...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='high_quality',
        time_limit=3600,
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5},
        num_stack_levels=1, # Enabled stacking for extra boost
        fit_strategy='sequential' # CRITICAL for Windows stability
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker()
