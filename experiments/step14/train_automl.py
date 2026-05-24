import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker_v14():
    train_path = Path("step14/train_ranker_v14.csv")
    df = pd.read_csv(train_path)
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step14/models/ranker_v6'
    
    print(f"Starting Step 14 Robust Training (Augmented 30k)...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='good_quality',
        time_limit=1800,
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker_v14()
