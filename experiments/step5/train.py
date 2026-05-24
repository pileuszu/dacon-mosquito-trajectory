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
from .model import MultiModalGMMTransformer
from .dataset import get_dataloaders
from utils.notifier import send_discord_notification

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gmm_nll_loss(batch_target, mu, log_sigma, pi):
    """
    batch_target: (batch, 3)
    mu: (batch, num_modes, 3)
    log_sigma: (batch, num_modes, 3)
    pi: (batch, num_modes)
    """
    # Reshape target to match mu
    target = batch_target.unsqueeze(1).expand_as(mu) # (batch, num_modes, 3)
    
    # Calculate log probability for each mode
    # log N(x | mu, sigma) = -0.5 * log(2pi) - log(sigma) - 0.5 * ((x-mu)/sigma)^2
    sigma = torch.exp(log_sigma)
    log_prob = -0.5 * ((target - mu) / sigma)**2 - log_sigma - 0.5 * np.log(2 * np.pi)
    
    # Sum over xyz dimensions
    log_prob = log_prob.sum(dim=2) # (batch, num_modes)
    
    # Combine with mode probabilities (pi)
    # log sum(pi * exp(log_prob))
    weighted_log_prob = torch.log(pi + 1e-10) + log_prob
    
    # Use logsumexp for stability
    loss = -torch.logsumexp(weighted_log_prob, dim=1)
    
    return loss.mean()

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
            "num_modes": NUM_MODES,
            "arch": "MultiModal-GMM-Transformer"
        }
    )
    
    # 2. Data & Model
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    model = MultiModalGMMTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        num_modes=NUM_MODES,
        dropout=DROPOUT
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    # Use Warm Restarts for jumping out of local minima
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=1e-6)

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
            target_res = item['residual'].to(device)
            
            optimizer.zero_grad()
            
            with amp.autocast('cuda'):
                final_pred, mu, log_sigma, pi = model(seq, cv_prior, last_pos, rot_mat)
                loss = gmm_nll_loss(target_res, mu, log_sigma, pi)
            
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # Slightly higher clip for GMM
            scaler.step(optimizer)
            scaler.update()
            
            train_losses.append(loss.item())
            pbar.set_postfix(loss=loss.item())

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
                    final_pred, mu, log_sigma, pi = model(seq, cv_prior, last_pos, rot_mat)
                    loss = gmm_nll_loss(target_res, mu, log_sigma, pi)
                
                val_losses.append(loss.item())
                dist = torch.norm(target - final_pred, dim=1)
                dist_errors.extend(dist.cpu().numpy())
                hit_1cm += (dist < 0.01).sum().item()
                hit_3cm += (dist < 0.03).sum().item()

        avg_val_loss = np.mean(val_losses)
        avg_dist_error = np.mean(dist_errors)
        hr_1cm = hit_1cm / len(val_loader.dataset)
        hr_3cm = hit_3cm / len(val_loader.dataset)

        # ETC
        epoch_duration = time.time() - epoch_start
        avg_epoch_time = epoch_duration # Simplified for step5
        remaining_time_sec = (EPOCHS - (epoch + 1)) * avg_epoch_time
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
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'step5_best_model.pth')
            print(f"--> Saved Best Model")
            early_stop_count = 0
        else:
            early_stop_count += 1
            if early_stop_count >= PATIENCE: break

    wandb.finish()
    total_dur = str(timedelta(seconds=int(time.time() - start_time)))
    final_msg = f"✅ **Step 5 (GMM) Complete!**\n- Best HR@1cm: `{hr_1cm:.4f}`\n- Duration: `{total_dur}`\n- Mention: @z5r10"
    send_discord_notification(DISCORD_WEBHOOK_URL, final_msg)

if __name__ == "__main__":
    train()
