import pandas as pd
import numpy as np
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker():
    train_path = Path("step10/train_ranker.csv")
    df = pd.read_csv(train_path)
    
    # Drop unnecessary columns
    # We KEEP cand_idx because it's a good feature for bias
    train_data = df.drop(columns=['id', 'reg_target'])
    
    label = 'target'
    save_path = 'step10/models/ranker'
    
    print(f"Starting Step 10 AutoML Training on {len(train_data)} rows...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='best_quality', # Deep stacking and ensembles
        time_limit=3600,
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker()
