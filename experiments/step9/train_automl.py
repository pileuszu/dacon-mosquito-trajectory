import pandas as pd
from autogluon.tabular import TabularPredictor
from pathlib import Path

def train_ranker():
    train_path = Path("step9/train_ranker_balanced.csv")
    df = pd.read_csv(train_path)
    
    # Drop unnecessary columns
    train_data = df.drop(columns=['id', 'reg_target'])
    
    label = 'target'
    save_path = 'step9/models/ranker'
    
    print(f"Starting AutoML Training on {len(train_data)} rows...")
    predictor = TabularPredictor(
        label=label, 
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2
    ).fit(
        train_data,
        presets='high_quality', # Balanced speed and quality
        time_limit=3600, # 1 hour limit
        num_stack_levels=1, # Add stacking for better ranking
        num_bag_folds=5
    )
    
    print("\nTraining Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    train_ranker()
