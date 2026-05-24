import pandas as pd
from pathlib import Path
from autogluon.tabular import TabularPredictor

def train_ranker_v13():
    train_path = Path("step13/train_ranker_v13.csv")
    df = pd.read_csv(train_path)
    # Drop non-feature columns
    train_data = df.drop(columns=['id'])
    
    label = 'target'
    save_path = 'step13/models/ranker_v5'
    
    print(f"Starting Step 13 Training (Extreme Precision V5)...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='good_quality', # Better than medium, enables some bagging/stacking
        time_limit=1800,
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5}
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker_v13()
