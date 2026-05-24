import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker_hybrid():
    train_path = Path("step16/train_ranker_v16.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step16/models/ranker_v16'
    
    print(f"Starting Step 16 HYBRID Training (Best Quality, 3 Hours)...")
    
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='best_quality',
        time_limit=10800, # 3 Hours
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nHYBRID Training Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker_hybrid()
