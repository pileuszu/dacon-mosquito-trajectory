import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from pathlib import Path

from .config import *
from .dataset import get_dataloaders
from .model import CandidateTransformerSelector

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_dataloaders(BATCH_SIZE)
    
    model = CandidateTransformerSelector(
        input_dim=12, 
        num_candidates=NUM_CANDIDATES,
        d_model=D_MODEL,
        dropout=DROPOUT
    ).to(device)
    
    # Ensure models directory exists
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / "best_model.pth"
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LEARNING_RATE, 
        steps_per_epoch=len(train_loader), epochs=EPOCHS
    )
    scaler = GradScaler()
    
    # Loss functions
    criterion_ce = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    criterion_mse = nn.MSELoss()
    
    best_hit_rate = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        
        for batch in pbar:
            seq = batch['seq'].to(device)
            candidates = batch['candidates'].to(device)
            cand_feats = batch['cand_feats'].to(device)
            prob_targets = batch['prob_targets'].to(device)
            best_idx = batch['best_cand_idx'].to(device)
            target = batch['target'].to(device)
            rot_mat = batch['rot_mat'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Forward pass
                logits, correction = model(seq, candidates, cand_feats, rot_mat)
                
                # 1. Selection Loss (Soft + Hard)
                loss_kl = criterion_kl(torch.log_softmax(logits, dim=-1), prob_targets)
                loss_ce = criterion_ce(logits, best_idx)
                
                # 2. Correction Loss (MSE in Global Space)
                # Final prediction = Selected Candidate + Rotated Correction
                with torch.no_grad():
                    pred_idx = logits.argmax(dim=-1)
                    chosen_cand = torch.stack([candidates[i, pred_idx[i]] for i in range(len(pred_idx))])
                
                # We want: Target = chosen_cand + rotated_correction
                # So: rotated_correction = Target - chosen_cand
                # Note: Model already outputs rotated correction internally if implemented that way, 
                # but let's assume correction is in local space and model rotates it.
                # If model(seq, candidates, rot_mat) returns (logits, final_refined_pred):
                final_pred = chosen_cand + correction # model.py should handle rotation
                loss_mse = criterion_mse(final_pred, target)
                
                loss = loss_kl * 10.0 + loss_ce * 1.0 + loss_mse * 100.0
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # Validation
        model.eval()
        val_hits = 0
        oracle_hits = 0
        total = 0
        val_dist_error = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="[Val]"):
                seq = batch['seq'].to(device)
                candidates = batch['candidates'].to(device)
                cand_feats = batch['cand_feats'].to(device)
                target = batch['target'].to(device)
                rot_mat = batch['rot_mat'].to(device)
                oracle_hit = batch['oracle_hit'].to(device)
                
                logits, correction = model(seq, candidates, cand_feats, rot_mat)
                pred_idx = logits.argmax(dim=-1)
                
                chosen_cand = torch.stack([candidates[i, pred_idx[i]] for i in range(len(pred_idx))])
                final_pred = chosen_cand + correction
                
                dists = torch.linalg.norm(final_pred - target, dim=1)
                val_hits += (dists <= R_HIT).sum().item()
                oracle_hits += oracle_hit.sum().item()
                val_dist_error += dists.mean().item()
                total += len(target)
                
        hit_rate = val_hits / total
        oracle_rate = oracle_hits / total
        avg_dist = val_dist_error / len(val_loader)
        
        print(f"Epoch {epoch+1}: Val Hit Rate = {hit_rate:.4f} | Oracle Rate = {oracle_rate:.4f} | Avg Dist = {avg_dist:.4f}cm")
        
        if hit_rate > best_hit_rate:
            best_hit_rate = hit_rate
            torch.save(model.state_dict(), model_save_path)
            print(f"--- Best Model Saved (Hit Rate: {best_hit_rate:.4f}) ---")

if __name__ == "__main__":
    train()
