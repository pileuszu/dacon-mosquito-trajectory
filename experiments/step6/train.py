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
from .model import TriAxisDiscreteTransformer
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

    # 1. Data & Model
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = TriAxisDiscreteTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_bins=NUM_BINS,
        dropout=DROPOUT
    ).to(device)
    
    # 2. WandB
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "lr": LEARNING_RATE,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "input_size": INPUT_SIZE,
            "arch": "TriAxis-Discrete-Physics-OneCycle"
        }
    )

    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.HuberLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # ⚡ OneCycleLR: Warm-up then Annealing for deep convergence
    total_steps = EPOCHS * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=LEARNING_RATE, 
        total_steps=total_steps,
        pct_start=0.1, # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1000.0
    )

    scaler = amp.GradScaler('cuda')

    best_val_loss = float('inf')
    early_stop_count = 0
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
            bin_labels = item['bin_labels'].to(device)
            offsets = item['offsets'].to(device)
            
            optimizer.zero_grad()
            
            with amp.autocast('cuda'):
                final_pred, logits, preds_off = model(seq, cv_prior, last_pos, rot_mat)
                loss_cls = criterion_cls(logits[0], bin_labels[:, 0]) + \
                           criterion_cls(logits[1], bin_labels[:, 1]) + \
                           criterion_cls(logits[2], bin_labels[:, 2])
                loss_reg = criterion_reg(preds_off[0].squeeze(), offsets[:, 0]) + \
                           criterion_reg(preds_off[1].squeeze(), offsets[:, 1]) + \
                           criterion_reg(preds_off[2].squeeze(), offsets[:, 2])
                total_loss = loss_cls + 1.0 * loss_reg
            
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            # ⚡ OneCycleLR steps after every batch
            scheduler.step()
            
            train_losses.append(total_loss.item())
            pbar.set_postfix(loss=total_loss.item(), lr=optimizer.param_groups[0]['lr'])

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
                target = item['target'].to(device); bin_labels = item['bin_labels'].to(device)
                offsets = item['offsets'].to(device)

                with amp.autocast('cuda'):
                    final_pred, logits, preds_off = model(seq, cv_prior, last_pos, rot_mat)
                    loss_cls = criterion_cls(logits[0], bin_labels[:, 0]) + \
                               criterion_cls(logits[1], bin_labels[:, 1]) + \
                               criterion_cls(logits[2], bin_labels[:, 2])
                    loss_reg = criterion_reg(preds_off[0].squeeze(), offsets[:, 0]) + \
                               criterion_reg(preds_off[1].squeeze(), offsets[:, 1]) + \
                               criterion_reg(preds_off[2].squeeze(), offsets[:, 2])
                    loss = loss_cls + loss_reg
                
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
        remaining_time_sec = (EPOCHS - (epoch + 1)) * epoch_duration
        etc_time = datetime.now() + timedelta(seconds=remaining_time_sec) - timedelta(hours=1)
        etc_str = etc_time.strftime("%H:%M:%S")

        wandb.log({
            "val/loss": avg_val_loss, 
            "val/hit_rate@1cm": hr_1cm, 
            "val/hit_rate@3cm": hr_3cm,
            "val/mean_dist_error": avg_dist_error,
            "train/lr": optimizer.param_groups[0]['lr'],
            "val/epoch": epoch
        })
        
        print(f"Epoch {epoch+1} - Val Loss: {avg_val_loss:.6f}, HR@1cm: {hr_1cm:.4f} | ETC: {etc_str}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'step6_best_model.pth')
            print(f"--> Saved Best Model")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE: break

    wandb.finish()
    total_dur = str(timedelta(seconds=int(time.time() - start_time)))
    final_msg = f"✅ **Step 6 (Discrete Physics) Complete!**\n- Best HR@1cm: `{hr_1cm:.4f}`\n- Duration: `{total_dur}`\n- Mention: @z5r10"
    send_discord_notification(DISCORD_WEBHOOK_URL, final_msg)

if __name__ == "__main__":
    train()
