import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import numpy as np
import random
import os
import pandas as pd
from datetime import datetime
from config import (
    BATCH_SIZE, LEARNING_RATE, EPOCHS, HIDDEN_SIZE, NUM_LAYERS,
    WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME, WANDB_GROUP, R_HIT, R_HIT_3CM, R_HIT_5CM,
    RANDOM_STATE, TRAIN_LABELS_PATH
)
from model import LSTMResidualModel
from dataset import get_dataloaders

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train():
    # 0. Set Seed
    set_seed(RANDOM_STATE)
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Dataloaders
    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # 3. Model, Loss, Optimizer, Scheduler
    model = LSTMResidualModel(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 4. WandB Init
    run_name = f"{WANDB_NAME}-{datetime.now().strftime('%m%d-%H%M%S')}"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        group=WANDB_GROUP,
        job_type="train",
        config={
            "batch_size": BATCH_SIZE,
            "lr": LEARNING_RATE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "optimizer": "AdamW",
            "loss": "SmoothL1",
            "scheduler": "ReduceLROnPlateau",
            "seed": RANDOM_STATE
        }
    )
    
    # Define metrics with prefixes to eliminate 'Charts' section
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="val/epoch")

    # 5. Training Loop
    global_step = 0
    best_val_hit = -1.0
    early_stop_patience = 10
    early_stop_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_losses = []
        
        for batch in train_loader:
            seq = batch["seq"].to(device)
            cv_prior = batch["cv_prior"].to(device)
            last_pos = batch["last_pos"].to(device)
            target_residual = batch["residual"].to(device)
            
            optimizer.zero_grad()
            _, pred_residual = model(seq, cv_prior, last_pos)
            
            loss = criterion(pred_residual, target_residual)
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
            # Log batch loss and LR
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]['lr'],
                    "train/step": global_step
                })
                
            global_step += 1
        
        avg_train_loss = np.mean(epoch_losses)
        
        # 6. Evaluation (Validation)
        model.eval()
        val_losses = []
        val_distances = []
        with torch.no_grad():
            for batch in test_loader:
                seq = batch["seq"].to(device)
                cv_prior = batch["cv_prior"].to(device)
                last_pos = batch["last_pos"].to(device)
                target_residual = batch["residual"].to(device)
                target_pos = batch["target"].to(device)
                
                final_pred, pred_residual = model(seq, cv_prior, last_pos)
                
                v_loss = criterion(pred_residual, target_residual)
                val_losses.append(v_loss.item())
                
                dist = torch.norm(final_pred - target_pos, dim=1)
                val_distances.extend(dist.cpu().numpy())
        
        avg_val_loss = np.mean(val_losses)
        val_distances = np.array(val_distances)
        hit_1cm = np.mean(val_distances <= R_HIT)
        hit_3cm = np.mean(val_distances <= R_HIT_3CM)
        mean_dist_err = np.mean(val_distances)
        
        # 7. Log Validation Metrics
        wandb.log({
            "val/epoch": epoch,
            "val/hit_rate@1cm": hit_1cm,
            "val/hit_rate@3cm": hit_3cm,
            "val/mean_dist_error": mean_dist_err,
            "val/loss": avg_val_loss,
            "val/train_loss": avg_train_loss
        })
        
        scheduler.step(avg_val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | Val Hit@1cm: {hit_1cm:.4f}")

        # 8. Save Best Model and Early Stopping
        if hit_1cm > best_val_hit:
            best_val_hit = hit_1cm
            torch.save(model.state_dict(), "step1_best_model.pth")
            print(f"  [New Best] Model saved with Val Hit@1cm: {hit_1cm:.4f}")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= early_stop_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs.")
                break

    # 9. Final Test Evaluation with Best Model
    print("\nTraining finished. Evaluating best model on test split...")
    model.load_state_dict(torch.load("step1_best_model.pth", map_location=device))
    model.eval()
    
    test_distances = []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            cv_prior = batch["cv_prior"].to(device)
            last_pos = batch["last_pos"].to(device)
            target_pos = batch["target"].to(device)
            final_pred, _ = model(seq, cv_prior, last_pos)
            dist = torch.norm(final_pred - target_pos, dim=1)
            test_distances.extend(dist.cpu().numpy())
            
    test_distances = np.array(test_distances)
    test_metrics = {
        "test/hit_rate@1cm": np.mean(test_distances <= R_HIT),
        "test/hit_rate@3cm": np.mean(test_distances <= R_HIT_3CM),
        "test/hit_rate@5cm": np.mean(test_distances <= R_HIT_5CM),
        "test/mean_dist_error": np.mean(test_distances),
        "test/max_dist_error": np.max(test_distances),
        "test/rmse": np.sqrt(np.mean(test_distances**2)),
    }
    
    for k, v in test_metrics.items():
        wandb.run.summary[k] = v
    
    print("\n--- Final Test Results (Best Model) ---")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    wandb.finish()
    print("\nTraining and Final Evaluation records completed in WandB.")

if __name__ == "__main__":
    train()
