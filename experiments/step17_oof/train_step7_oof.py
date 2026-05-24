import pandas as pd
import numpy as np
import os
import sys
import traceback
from pathlib import Path
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import KFold

# Reuse Step 6/7 logic
sys.path.append(os.getcwd())
from step6.dataset import DiscreteDataset
from step6.config import *
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def prepare_tabular_data_v17(files, labels_df):
    dataset = DiscreteDataset(files, labels_df, augment=False)
    rows = []
    print(f"Transforming {len(dataset)} samples to tabular format...")
    for i in tqdm(range(len(dataset))):
        item = dataset[i]
        seq = item['seq'].numpy() 
        cv_prior = item['cv_prior'].numpy()
        fid = item['id']
        
        flat_features = seq.flatten()
        target_xyz = item['target'].numpy()
        last_xyz = item['last_pos'].numpy()
        rot_mat = item['rot_mat'].numpy()
        
        target_res_global = target_xyz - (last_xyz + (rot_mat.T @ cv_prior))
        
        row = {"id": fid}
        for f_idx in range(len(flat_features)):
            row[f'feat_{f_idx}'] = flat_features[f_idx]
        
        row['target_x'] = target_res_global[0]
        row['target_y'] = target_res_global[1]
        row['target_z'] = target_res_global[2]
        
        row['last_x'] = last_xyz[0]; row['last_y'] = last_xyz[1]; row['last_z'] = last_xyz[2]
        row['cv_x'] = (rot_mat.T @ cv_prior)[0]; row['cv_y'] = (rot_mat.T @ cv_prior)[1]; row['cv_z'] = (rot_mat.T @ cv_prior)[2]
        
        rows.append(row)
    return pd.DataFrame(rows)

def run_5fold_oof():
    try:
        send_discord_notification(URL, "🚀 [Step 17] 5-Fold OOF Training Started...")
        
        labels_df = pd.read_csv(TRAIN_LABELS_PATH)
        all_ids = labels_df['id'].unique()
        all_files = [TRAIN_DIR / f"{fid}.csv" for fid in all_ids]
        
        full_df = prepare_tabular_data_v17(all_files, labels_df)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        oof_results = []
        targets = ['target_x', 'target_y', 'target_z']
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
            train_data = full_df.iloc[train_idx]
            val_data = full_df.iloc[val_idx]
            fold_preds = val_data[['id', 'last_x', 'last_y', 'last_z', 'cv_x', 'cv_y', 'cv_z']].copy()
            
            for target in targets:
                save_path = f'step17_oof/models/fold{fold}/{target}'
                drop_cols = [t for t in targets if t != target] + ['id', 'last_x', 'last_y', 'last_z', 'cv_x', 'cv_y', 'cv_z']
                predictor = TabularPredictor(label=target, problem_type='regression', eval_metric='mean_absolute_error', path=save_path).fit(
                    train_data.drop(columns=drop_cols), presets='medium_quality', time_limit=300
                )
                fold_preds[f'pred_{target}'] = predictor.predict(val_data.drop(columns=drop_cols))
            
            oof_results.append(fold_preds)
            send_discord_notification(URL, f"📦 [Step 17] Fold {fold+1}/5 Training Complete.")
        
        final_oof = pd.concat(oof_results)
        final_oof['x'] = final_oof['last_x'] + final_oof['cv_x'] + final_oof['pred_target_x']
        final_oof['y'] = final_oof['last_y'] + final_oof['cv_y'] + final_oof['pred_target_y']
        final_oof['z'] = final_oof['last_z'] + final_oof['cv_z'] + final_oof['pred_target_z']
        
        out_path = "step17_oof/step7_oof_train.csv"
        final_oof[['id', 'x', 'y', 'z']].to_csv(out_path, index=False)
        send_discord_notification(URL, f"✅ [Step 17] OOF Training Successfully Finished! Data saved to {out_path}")

    except Exception as e:
        error_msg = f"❌ [Step 17] ERROR Occurred:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        raise e

if __name__ == "__main__":
    run_5fold_oof()
