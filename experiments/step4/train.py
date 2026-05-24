import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import random
import os
import time
from datetime import datetime, timedelta
from tqdm import tqdm
from torch import amp

from .config import *
from .model import BiomechanicalTransformer
from .dataset import get_dataloaders
from utils.notifier import send_discord_notification

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    set_seed(RANDOM_STATE)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "target_scale": TARGET_SCALE,
            "arch": "Biomechanical-Transformer-AMP"
        }
    )
    
    # Define metrics for better plotting
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")

    # 2. Data & Model
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = BiomechanicalTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(device)
    
    criterion_res = nn.SmoothL1Loss(reduction='none') 
    criterion_state = nn.BCEWithLogitsLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # ⚡ Cosine Annealing with Warm Restarts to shake local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=1, eta_min=1e-6)

    # ⚡ New AMP Scaler syntax
    scaler = amp.GradScaler('cuda')

    best_val_loss = float('inf')
    early_stop_count = 0
    global_step = 0
    epoch_times = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for item in pbar:
            seq = item['seq'].to(device)
            cv_prior = item['cv_prior'].to(device)
            last_pos = item['last_pos'].to(device)
            rot_mat = item['rot_mat'].to(device)
            target_res = item['residual'].to(device)
            
            curvatures = seq[:, -1, 9] 
            saccade_gt = (curvatures > 5.0).float().unsqueeze(-1)
            
            optimizer.zero_grad()
            
            # ⚡ New amp.autocast syntax
            with amp.autocast('cuda'):
                final_pred, pred_res, pred_state = model(seq, cv_prior, last_pos, rot_mat)
                res_loss_raw = criterion_res(pred_res, target_res).mean(dim=1, keepdim=True)
                weights = 1.0 + saccade_gt 
                res_loss = (res_loss_raw * weights).mean()
                state_loss = criterion_state(pred_state, saccade_gt)
                total_loss = res_loss + 0.1 * state_loss
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(total_loss.item())
            global_step += 1
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss": total_loss.item(), 
                    "train/lr": optimizer.param_groups[0]['lr'], 
                    "train/step": global_step
                })
            pbar.set_postfix(loss=total_loss.item())

        # Validation
        model.eval()
        val_losses = []
        dist_errors = []
        hit_1cm = 0
        hit_3cm = 0
        
        with torch.no_grad():
            for item in val_loader:
                seq = item['seq'].to(device); cv_prior = item['cv_prior'].to(device)
                last_pos = item['last_pos'].to(device); rot_mat = item['rot_mat'].to(device)
                target = item['target'].to(device); target_res = item['residual'].to(device)

                with amp.autocast('cuda'):
                    final_pred, pred_res, _ = model(seq, cv_prior, last_pos, rot_mat)
                    loss = criterion_res(pred_res, target_res).mean()
                
                val_losses.append(loss.item())
                dist = torch.norm(target - final_pred, dim=1)
                dist_errors.extend(dist.cpu().numpy())
                hit_1cm += (dist < 0.01).sum().item()
                hit_3cm += (dist < 0.03).sum().item()

        avg_val_loss = np.mean(val_losses)
        avg_dist_error = np.mean(dist_errors)
        hr_1cm = hit_1cm / len(val_loader.dataset)
        hr_3cm = hit_3cm / len(val_loader.dataset)

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = np.mean(epoch_times)
        remaining_time_sec = (EPOCHS - (epoch + 1)) * avg_epoch_time
        etc_time = datetime.now() + timedelta(seconds=remaining_time_sec) - timedelta(hours=1)
        etc_str = etc_time.strftime("%H:%M:%S")

        wandb.log({
            "val/loss": avg_val_loss, 
            "val/hit_rate@1cm": hr_1cm, 
            "val/hit_rate@3cm": hr_3cm,
            "val/mean_dist_error": avg_dist_error,
            "val/epoch": epoch
        })
        
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.6f}, HR@1cm: {hr_1cm:.4f} | ETC: {etc_str}")
        
        # ⚡ Step by epoch for CosineAnnealingWarmRestarts
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'step4_best_model.pth')
            print(f"--> Saved Best Model")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE: break

    wandb.finish()
    total_dur = str(timedelta(seconds=int(time.time() - start_time)))
    final_msg = f"✅ **Step 4 Complete!**\n- Run: `{WANDB_RUN_NAME}`\n- HR@1cm: `{hr_1cm:.4f}`\n- Duration: `{total_dur}`\n- Mention: @z5r10"
    send_discord_notification(DISCORD_WEBHOOK_URL, final_msg)
    print(f"Training Complete. Duration: {total_dur}")

if __name__ == "__main__":
    train()
