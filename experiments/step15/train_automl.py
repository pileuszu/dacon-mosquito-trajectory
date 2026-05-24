import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker_ultimate():
    train_path = Path("step15/train_ranker_v15.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step15/models/ranker_v7_ultimate'
    
    print(f"Starting Step 15 ULTIMATE Training (Best Quality, 4 Hours)...")
    print(f"Dataset size: {len(train_data)} rows")
    
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='best_quality', # Maximum power
        time_limit=14400,        # 4 Hours
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nULTIMATE Training Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker_ultimate()
