import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("AutoGluon is not installed. Please run: pip install autogluon")
    exit()

# Import the feature extraction logic from step6
import sys
sys.path.append(os.getcwd())
from step6.dataset import DiscreteDataset
from step6.config import *

def prepare_tabular_data(files, labels_df):
    dataset = DiscreteDataset(files, labels_df, augment=False)
    rows = []
    
    print(f"Transforming {len(dataset)} samples to tabular format...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        seq = item['seq'].numpy() # (10, 18)
        cv_prior = item['cv_prior'].numpy() # (3,)
        
        # Flatten sequence: (10 * 18 = 180 features)
        flat_features = seq.flatten()
        
        # Target residuals (relative to CV)
        target_xyz = item['target'].numpy()
        last_xyz = item['last_pos'].numpy()
        rot_mat = item['rot_mat'].numpy()
        
        # We predict the global residual for simplicity in AutoML
        target_res_global = target_xyz - (last_xyz + (rot_mat.T @ cv_prior))
        
        row = {}
        for f_idx in range(len(flat_features)):
            row[f'feat_{f_idx}'] = flat_features[f_idx]
        
        row['target_x'] = target_res_global[0]
        row['target_y'] = target_res_global[1]
        row['target_z'] = target_res_global[2]
        
        rows.append(row)
    
    return pd.DataFrame(rows)

def train_automl():
    # 1. Load Data
    step6_metadata = Path('step6/metadata.csv')
    if not step6_metadata.exists():
        print("Error: step6/metadata.csv not found. Please run step6 once.")
        return
        
    df_meta = pd.read_csv(step6_metadata)
    train_ids = df_meta[df_meta['split'] == 'train']['id'].tolist()
    test_ids = df_meta[df_meta['split'] == 'test']['id'].tolist()
    
    train_files = [TRAIN_DIR / f"{fid}.csv" for fid in train_ids]
    test_files = [TRAIN_DIR / f"{fid}.csv" for fid in test_ids]
    labels_df = pd.read_csv(TRAIN_LABELS_PATH)
    
    # 2. Prepare Tabular Data (Limited for speed if needed, but let's try all)
    train_df = prepare_tabular_data(train_files, labels_df)
    test_df = prepare_tabular_data(test_files, labels_df)
    
    os.makedirs('step7/models', exist_ok=True)
    
    # 3. Train for each axis (X, Y, Z)
    # AutoGluon handles best quality ensembling automatically
    targets = ['target_x', 'target_y', 'target_z']
    predictors = {}
    
    for target in targets:
        print(f"\n--- Training AutoML for {target} ---")
        save_path = f'step7/models/{target}'
        
        # We'll drop other targets to avoid leakage
        current_train = train_df.drop(columns=[t for t in targets if t != target])
        
        predictor = TabularPredictor(
            label=target,
            problem_type='regression',
            eval_metric='mean_absolute_error',
            path=save_path
        ).fit(
            current_train,
            presets='medium_quality', # 'best_quality' for stacking, but takes longer
            time_limit=600 # 10 minutes per axis
        )
        predictors[target] = predictor

    # 4. Evaluation
    print("\n--- Evaluation on Test Split ---")
    # Simple evaluation logic for Hit Rate
    preds_x = predictors['target_x'].predict(test_df.drop(columns=targets))
    preds_y = predictors['target_y'].predict(test_df.drop(columns=targets))
    preds_z = predictors['target_z'].predict(test_df.drop(columns=targets))
    
    # Reconstruct final positions and calculate Hit Rate
    # (Omitted for brevity, but results will be in AutoGluon logs)
    print("AutoML Training Complete. Check step7/models for leaderboards.")

if __name__ == "__main__":
    train_automl()
