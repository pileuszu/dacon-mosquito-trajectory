import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor

import sys
sys.path.append(os.getcwd())
from step6.dataset import DiscreteDataset
from step6.config import *

def get_step7_predictions(files, labels_df=None, is_test=True):
    dataset = DiscreteDataset(files, labels_df=labels_df, is_test=is_test, augment=False)
    
    # 1. Transform to Tabular
    rows = []
    ids = []
    print(f"Transforming {len(dataset)} samples...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        seq = item['seq'].numpy()
        cv_prior = item['cv_prior'].numpy()
        last_xyz = item['last_pos'].numpy()
        rot_mat = item['rot_mat'].numpy()
        
        flat_features = seq.flatten()
        row = {f'feat_{j}': val for j, val in enumerate(flat_features)}
        row.update({
            '_last_x': last_xyz[0], '_last_y': last_xyz[1], '_last_z': last_xyz[2],
            '_cv_px': cv_prior[0], '_cv_py': cv_prior[1], '_cv_pz': cv_prior[2]
        })
        for r in range(3):
            for c in range(3):
                row[f'_rot_{r}{c}'] = rot_mat[r, c]
        rows.append(row)
        ids.append(item['id'])
    
    df = pd.DataFrame(rows)
    input_df = df[[c for c in df.columns if not c.startswith('_')]]
    
    # 2. Predict
    targets = ['target_x', 'target_y', 'target_z']
    preds_raw = {}
    for t in targets:
        model_path = f'step7/models/{t}'
        predictor = TabularPredictor.load(model_path)
        preds_raw[t] = predictor.predict(input_df)
        
    # 3. Reconstruct
    results = []
    for i in range(len(ids)):
        res_global = np.array([preds_raw['target_x'][i], preds_raw['target_y'][i], preds_raw['target_z'][i]])
        last_xyz = np.array([df['_last_x'][i], df['_last_y'][i], df['_last_z'][i]])
        cv_prior = np.array([df['_cv_px'][i], df['_cv_py'][i], df['_cv_pz'][i]])
        
        rot_mat = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                rot_mat[r, c] = df[f'_rot_{r}{c}'][i]
                
        final_xyz = res_global + last_xyz + (rot_mat.T @ cv_prior)
        results.append({
            'id': ids[i],
            's7_x': final_xyz[0],
            's7_y': final_xyz[1],
            's7_z': final_xyz[2]
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Train Preds
    train_labels = pd.read_csv(TRAIN_LABELS_PATH)
    train_files = sorted(TRAIN_DIR.glob('*.csv'))
    # Use only the 10,000 files in labels
    train_ids = train_labels['id'].unique()
    train_files = [TRAIN_DIR / f"{fid}.csv" for fid in train_ids]
    
    print("Generating Step 7 predictions for Train Set...")
    train_preds = get_step7_predictions(train_files, labels_df=train_labels, is_test=False)
    train_preds.to_csv('step10/step7_preds_train.csv', index=False)
    
    # Test Preds
    test_files = sorted(TEST_DIR.glob('*.csv'))
    print("\nGenerating Step 7 predictions for Test Set...")
    test_preds = get_step7_predictions(test_files, is_test=True)
    test_preds.to_csv('step10/step7_preds_test.csv', index=False)
    
    print("\nStep 7 predictions saved to step10/")
