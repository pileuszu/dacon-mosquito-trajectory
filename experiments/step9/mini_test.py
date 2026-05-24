import pandas as pd
import numpy as np
from pathlib import Path
from step9.prepare_data import prepare_tabular_data
from autogluon.tabular import TabularPredictor

def mini_test():
    # 1. Prepare small data (100 samples)
    prepare_tabular_data(limit=100)
    
    df = pd.read_csv("step9/train_ranker.csv")
    train_data = df.drop(columns=['id', 'reg_target'])
    
    # 2. Train with minimal settings
    predictor = TabularPredictor(
        label='target',
        eval_metric='roc_auc'
    ).fit(
        train_data,
        time_limit=300,
        presets='medium_quality', # No stacking/bagging
        ag_args_fit={'ag.max_memory_usage_ratio': 1.5} # Allow more memory
    )
    
    print("\nMini Test Complete!")
    print(predictor.leaderboard())

if __name__ == "__main__":
    mini_test()
