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

from .config import *
from .model import LSTMHybridModel
from .dataset import get_dataloaders
from utils.notifier import send_discord_notification

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1504302314620715042/QqgM9VI4Z-o9IqV10khxjToRfcSR-WORkHkO7srYBo4C5ZjYlRFGVGChDA0WBUjyxgR7"

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

    # 1. Initialize WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "loss_fn": "SmoothL1Loss",
            "features": "xyz_v_a"
        }
    )
    
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/epoch")
    wandb.define_metric("val/*", step_metric="val/epoch")

    # 2. Data & Model
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = LSTMHybridModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    
    criterion = nn.SmoothL1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 3. Training Loop
    best_val_loss = float('inf')
    early_stop_count = 0
    global_step = 0
    
    epoch_times = []
    start_time = time.time()

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()
        train_losses = []
        
        # tqdm description for remaining time
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for item in pbar:
            seq = item['seq'].to(device)
            cv_prior = item['cv_prior'].to(device)
            last_pos = item['last_pos'].to(device)
            target = item['target'].to(device)
            residual_gt = item['residual'].to(device)
            
            optimizer.zero_grad()
            final_pred, pred_residual = model(seq, cv_prior, last_pos)
            loss = criterion(pred_residual, residual_gt)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            global_step += 1
            
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/step": global_step
                })
            
            pbar.set_postfix(loss=loss.item())

        # Validation
        model.eval()
        val_losses = []
        dist_errors = []
        hit_1cm = 0
        hit_3cm = 0
        
        with torch.no_grad():
            for item in val_loader:
                seq = item['seq'].to(device)
                cv_prior = item['cv_prior'].to(device)
                last_pos = item['last_pos'].to(device)
                target = item['target'].to(device)
                residual_gt = item['residual'].to(device)

                final_pred, pred_residual = model(seq, cv_prior, last_pos)
                loss = criterion(pred_residual, residual_gt)
                val_losses.append(loss.item())
                
                dist = torch.norm(target - final_pred, dim=1)
                dist_errors.extend(dist.cpu().numpy())
                hit_1cm += (dist < 0.01).sum().item()
                hit_3cm += (dist < 0.03).sum().item()

        avg_val_loss = np.mean(val_losses)
        avg_dist_error = np.mean(dist_errors)
        hr_1cm = hit_1cm / len(val_loader.dataset)
        hr_3cm = hit_3cm / len(val_loader.dataset)

        # Calculate ETC
        epoch_end = time.time()
        epoch_duration = epoch_end - epoch_start
        epoch_times.append(epoch_duration)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = EPOCHS - (epoch + 1)
        remaining_time_sec = remaining_epochs * avg_epoch_time
        
        etc_time = datetime.now() + timedelta(seconds=remaining_time_sec) - timedelta(hours=1)
        etc_str = etc_time.strftime("%H:%M:%S")

        wandb.log({
            "val/loss": avg_val_loss,
            "val/mean_dist_error": avg_dist_error,
            "val/hit_rate@1cm": hr_1cm,
            "val/hit_rate@3cm": hr_3cm,
            "val/epoch": epoch,
            "val/etc_seconds": remaining_time_sec
        })

        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.6f}, HR@1cm: {hr_1cm:.4f} | ETC: {etc_str}")

        # Learning Rate Scheduler
        scheduler.step(avg_val_loss)

        # Early Stopping & Model Save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'step2_best_model.pth')
            print(f"--> Saved Best Model (Val Loss: {best_val_loss:.6f})")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    wandb.finish()
    
    # Send Discord Notification
    total_duration = time.time() - start_time
    duration_str = str(timedelta(seconds=int(total_duration)))
    final_msg = (f"✅ **Step 2 Training Complete!**\n"
                 f"- Run Name: `{WANDB_RUN_NAME}`\n"
                 f"- Best Val Loss: `{best_val_loss:.6f}`\n"
                 f"- Total Duration: `{duration_str}`\n"
                 f"- Mention: @z5r10")
    send_discord_notification(DISCORD_WEBHOOK_URL, final_msg)
    
    print(f"Training Complete. Total Duration: {duration_str}")

if __name__ == "__main__":
    train()
