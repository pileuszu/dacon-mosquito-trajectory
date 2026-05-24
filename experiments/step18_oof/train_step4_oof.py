import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import sys
import traceback
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold

sys.path.append(os.getcwd())
from step4.config import *
from step4.model import BiomechanicalTransformer
from step4.dataset import BiomechanicalDataset
from utils.notifier import send_discord_notification

URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

def run_step4_oof():
    try:
        send_discord_notification(URL, "🚀 [Step 18] Step 4 (EqMotion) 5-Fold OOF Training Started on CPU...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load all train data
        labels_df = pd.read_csv(TRAIN_LABELS_PATH)
        all_ids = labels_df['id'].unique()
        all_files = [TRAIN_DIR / f"{fid}.csv" for fid in all_ids]
        
        # Restored original params for GPU training
        EPOCHS_GPU = 150
        PATIENCE_GPU = 20
        BATCH_SIZE_GPU = 2048
        
        dataset = BiomechanicalDataset(all_files, labels_df)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        
        oof_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
            print(f"\n===== FOLD {fold+1} / 5 =====")
            
            # Subsets
            train_sub = torch.utils.data.Subset(dataset, train_idx)
            val_sub = torch.utils.data.Subset(dataset, val_idx)
            
            train_loader = torch.utils.data.DataLoader(train_sub, batch_size=BATCH_SIZE_GPU, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_sub, batch_size=BATCH_SIZE_GPU, shuffle=False)
            
            model = BiomechanicalTransformer(
                input_size=INPUT_SIZE, d_model=D_MODEL, nhead=NHEAD,
                num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD, dropout=DROPOUT
            ).to(device)
            
            criterion_res = nn.SmoothL1Loss(reduction='none') 
            criterion_state = nn.BCEWithLogitsLoss()
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)
            
            best_val_loss = float('inf')
            early_stop_count = 0
            
            for epoch in range(EPOCHS_GPU):
                model.train()
                for item in train_loader:
                    seq = item['seq'].to(device); cv_prior = item['cv_prior'].to(device)
                    last_pos = item['last_pos'].to(device); rot_mat = item['rot_mat'].to(device)
                    target_res = item['residual'].to(device)
                    
                    curvatures = seq[:, -1, 9] 
                    saccade_gt = (curvatures > 5.0).float().unsqueeze(-1)
                    
                    optimizer.zero_grad()
                    final_pred, pred_res, pred_state = model(seq, cv_prior, last_pos, rot_mat)
                    
                    res_loss_raw = criterion_res(pred_res, target_res).mean(dim=1, keepdim=True)
                    weights = 1.0 + saccade_gt 
                    res_loss = (res_loss_raw * weights).mean()
                    state_loss = criterion_state(pred_state, saccade_gt)
                    
                    total_loss = res_loss + 0.1 * state_loss
                    total_loss.backward()
                    optimizer.step()
                
                # Validation
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for item in val_loader:
                        seq = item['seq'].to(device); cv_prior = item['cv_prior'].to(device)
                        last_pos = item['last_pos'].to(device); rot_mat = item['rot_mat'].to(device)
                        target_res = item['residual'].to(device)
                        
                        _, pred_res, _ = model(seq, cv_prior, last_pos, rot_mat)
                        loss = criterion_res(pred_res, target_res).mean()
                        val_losses.append(loss.item())
                
                avg_val_loss = np.mean(val_losses)
                scheduler.step()
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_state = model.state_dict()
                    early_stop_count = 0
                else:
                    early_stop_count += 1
                    if early_stop_count >= PATIENCE_GPU: break
                    
            print(f"Fold {fold+1} Best Val Loss: {best_val_loss:.6f}")
            send_discord_notification(URL, f"📦 [Step 18] Fold {fold+1}/5 EqMotion Training Complete.")
            
            # Predict OOF
            model.load_state_dict(best_state)
            model.eval()
            fold_results = []
            with torch.no_grad():
                for i in val_idx:
                    item = dataset[i]
                    seq = item['seq'].unsqueeze(0).to(device)
                    cv_prior = item['cv_prior'].unsqueeze(0).to(device)
                    last_pos = item['last_pos'].unsqueeze(0).to(device)
                    rot_mat = item['rot_mat'].unsqueeze(0).to(device)
                    
                    final_pred, _, _ = model(seq, cv_prior, last_pos, rot_mat)
                    pos = final_pred.squeeze().cpu().numpy()
                    
                    fold_results.append({
                        "id": item['id'],
                        "x": pos[0],
                        "y": pos[1],
                        "z": pos[2]
                    })
            oof_results.extend(fold_results)
            
        out_df = pd.DataFrame(oof_results)
        out_path = Path("step18_oof/step4_oof_train.csv")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        send_discord_notification(URL, f"✅ [Step 18] EqMotion OOF Training Successfully Finished! Data saved to {out_path}")

    except Exception as e:
        error_msg = f"❌ [Step 18] ERROR Occurred:\n{str(e)}\n\n{traceback.format_exc()}"
        send_discord_notification(URL, error_msg)
        raise e

if __name__ == "__main__":
    run_step4_oof()
