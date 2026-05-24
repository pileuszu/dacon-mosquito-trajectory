import pandas as pd
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm

# 🛠 몽키 패치: AutoGluon의 패키지 메타데이터 버그 수정
import importlib.metadata
import autogluon.common.utils.utils as ag_utils

def robust_get_package_versions():
    package_version_dict = {}
    for dist in importlib.metadata.distributions():
        try:
            name = dist.metadata.get("Name")
            if name:
                package_version_dict[name.lower()] = dist.version
        except Exception:
            continue
    return package_version_dict

ag_utils.get_package_versions = robust_get_package_versions

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("AutoGluon is not installed. Please run: pip install autogluon")
    exit()

import sys
sys.path.append(os.getcwd())
from step6.dataset import DiscreteDataset
from step6.config import *

def prepare_test_tabular(files):
    # labels_df is not needed for test inference
    dataset = DiscreteDataset(files, labels_df=None, is_test=True, augment=False)
    rows = []
    ids = []
    
    print(f"Transforming {len(dataset)} test samples to tabular format...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        seq = item['seq'].numpy()
        cv_prior = item['cv_prior'].numpy()
        last_xyz = item['last_pos'].numpy()
        rot_mat = item['rot_mat'].numpy()
        
        flat_features = seq.flatten()
        
        row = {}
        for f_idx in range(len(flat_features)):
            row[f'feat_{f_idx}'] = flat_features[f_idx]
        
        # Store context for reconstruction
        row['_last_x'] = last_xyz[0]
        row['_last_y'] = last_xyz[1]
        row['_last_z'] = last_xyz[2]
        row['_cv_px'] = cv_prior[0]
        row['_cv_py'] = cv_prior[1]
        row['_cv_pz'] = cv_prior[2]
        
        # rot_mat elements
        for r in range(3):
            for c in range(3):
                row[f'_rot_{r}{c}'] = rot_mat[r, c]
        
        rows.append(row)
        ids.append(item['id'])
    
    return pd.DataFrame(rows), ids

def run_inference():
    # 1. Load Test Files
    test_files = sorted(TEST_DIR.glob('*.csv'))
    test_df, test_ids = prepare_test_tabular(test_files)
    
    # 2. Load Predictors
    targets = ['target_x', 'target_y', 'target_z']
    predictions = {}
    
    for target in targets:
        model_path = f'step7/models/{target}'
        if not os.path.exists(model_path):
            print(f"Error: Model for {target} not found at {model_path}!")
            return
        
        print(f"Loading predictor for {target}...")
        predictor = TabularPredictor.load(model_path)
        # Drop internal columns before prediction
        input_df = test_df[[c for c in test_df.columns if not c.startswith('_')]]
        predictions[target] = predictor.predict(input_df)

    # 3. Reconstruct Global Coordinates
    results = []
    print("Reconstructing global coordinates...")
    for i in range(len(test_ids)):
        # Predicted global residual relative to rotated CV prior
        res_global = np.array([
            predictions['target_x'][i],
            predictions['target_y'][i],
            predictions['target_z'][i]
        ])
        
        last_xyz = np.array([test_df['_last_x'][i], test_df['_last_y'][i], test_df['_last_z'][i]])
        cv_prior = np.array([test_df['_cv_px'][i], test_df['_cv_py'][i], test_df['_cv_pz'][i]])
        
        # Restore rotation matrix
        rot_mat = np.zeros((3, 3))
        for r in range(3):
            for c in range(3):
                rot_mat[r, c] = test_df[f'_rot_{r}{c}'][i]
        
        # Reconstruct Final: Last + Rot^T @ (CV_Prior + Residual_Rot)
        # In step7 training, we defined target_res_global = target_xyz - (last_xyz + (rot_mat.T @ cv_prior))
        # So: target_xyz = res_global + last_xyz + (rot_mat.T @ cv_prior)
        
        final_xyz = res_global + last_xyz + (rot_mat.T @ cv_prior)
        
        results.append({
            'id': test_ids[i],
            'x': final_xyz[0],
            'y': final_xyz[1],
            'z': final_xyz[2]
        })

    # 4. Save Submission
    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    res_df = pd.DataFrame(results)
    final_df = submission_df[['id']].merge(res_df, on='id', how='left')
    
    os.makedirs('outputs/step7', exist_ok=True)
    final_df.to_csv('outputs/step7/submission.csv', index=False)
    print(f"AutoML Submission saved to outputs/step7/submission.csv")

if __name__ == "__main__":
    run_inference()
