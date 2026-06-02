import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import hashlib
import argparse
import traceback
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import SotaSpatiotemporalModel, extract_features, SotaFocalSoftHitLoss

class MosquitoCandidatesDataset(Dataset):
    def __init__(self, X, y, candidates):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.candidates = torch.tensor(candidates, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.candidates[idx]

def stable_fold_id(sample_id: str, folds: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    parser.add_argument("--noise-scale", type=float, default=0.01, help="Noise perturbation scale for CFM base point")
    args = parser.parse_args()
    
    send_discord_notification(
        None,
        f"🚀 Started: [Step 57 train_automl.py] Training 5-fold Spatiotemporal AI Model...\n"
        f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}"
    )
    
    try:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Load dataset
        data_dir = Path("step57_spatiotemporal_ai/data")
        X_train = np.load(data_dir / "train_x.npy")
        y_train = np.load(data_dir / "train_y.npy")
        candidates_train = np.load(data_dir / "train_candidates.npy")
        
        with open(data_dir / "train_ids.json", "r") as f:
            train_ids = json.load(f)
            
        print(f"Loaded train dataset. X_train: {X_train.shape}, candidates: {candidates_train.shape}")
        
        # Compute stable fold ids
        fold_ids = np.asarray([stable_fold_id(sid, args.folds) for sid in train_ids])
        
        oof_preds_1step = np.zeros_like(y_train)
        oof_preds_2step = np.zeros_like(y_train)
        
        models_dir = Path("step57_spatiotemporal_ai/models")
        models_dir.mkdir(exist_ok=True)
        
        fold_hr_scores_1step = []
        fold_hr_scores_2step = []
        
        for fold in range(args.folds):
            print(f"\n--- FOLD {fold + 1} / {args.folds} ---")
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            
            X_tr, y_tr, cand_tr = X_train[train_mask], y_train[train_mask], candidates_train[train_mask]
            X_val, y_val, cand_val = X_train[val_mask], y_train[val_mask], candidates_train[val_mask]
            
            train_ds = MosquitoCandidatesDataset(X_tr, y_tr, cand_tr)
            val_ds = MosquitoCandidatesDataset(X_val, y_val, cand_val)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
            
            # Compute feature scaling stats using training set
            train_coords_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, _, _, mean_stats, std_stats = extract_features(train_coords_tensor)
                
            # Save scaling stats
            stats_path = models_dir / f"stats_fold_{fold}.pt"
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, stats_path)
            
            # Create Model (Explicit Relational Branching + Spherical Dynamics + CFM)
            model = SotaSpatiotemporalModel(feature_dim=44, latent_dim=64, num_candidates=36, max_norm=0.06).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
            criterion = SotaFocalSoftHitLoss(delta=0.001026, alpha=400.0, target_dist=0.01).to(device)
            
            best_val_hr_2step = 0.0
            best_epoch = 0
            best_val_preds_1step = None
            best_val_preds_2step = None
            
            for epoch in range(1, args.epochs + 1):
                model.train()
                epoch_loss = 0.0
                epoch_cfm_loss = 0.0
                epoch_pred_loss = 0.0
                epoch_cons_loss = 0.0
                
                for Xb, yb, candb in train_loader:
                    Xb, yb, candb = Xb.to(device), yb.to(device), candb.to(device)
                    optimizer.zero_grad()
                    
                    # 1. Forward CFM flow with extended 44D feature map and spherical sequence
                    ft, df_c, df_s, _, _ = extract_features(Xb, mean_stats, std_stats)
                    pred_v, target_v, x_0, h_context = model.forward_flow(Xb, ft, df_c, df_s, candb, yb, noise_scale=args.noise_scale)
                    
                    # 2. CFM Vector Field Loss (Huber Loss)
                    loss_cfm = F.huber_loss(pred_v, target_v, delta=0.001)
                    
                    # 3. Predict coordinates with 1-step and 2-step integrations for consistency
                    # t=0 velocity
                    t0 = torch.zeros(Xb.shape[0], device=device)
                    v0 = model.vector_field(x_0, t0, h_context)
                    pred_1step = x_0 + v0
                    
                    # 2-step Euler integration
                    dt = 0.5
                    x_half = x_0 + v0 * dt
                    t05 = torch.ones(Xb.shape[0], device=device) * 0.5
                    v05 = model.vector_field(x_half, t05, h_context)
                    pred_2step = x_half + v05 * dt
                    
                    # 4. Focal Soft Hit + Huber loss on outputs
                    loss_pred_1step = criterion(pred_1step, yb)
                    loss_pred_2step = criterion(pred_2step, yb)
                    loss_pred = loss_pred_1step + loss_pred_2step
                    
                    # 5. Semigroup consistency loss
                    loss_cons = F.mse_loss(pred_1step, pred_2step)
                    
                    # 6. Combined loss
                    loss = 10.0 * loss_cfm + 1.0 * loss_pred + 50.0 * loss_cons
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * len(Xb)
                    epoch_cfm_loss += loss_cfm.item() * len(Xb)
                    epoch_pred_loss += loss_pred.item() * len(Xb)
                    epoch_cons_loss += loss_cons.item() * len(Xb)
                    
                scheduler.step()
                epoch_loss /= len(train_loader.dataset)
                epoch_cfm_loss /= len(train_loader.dataset)
                epoch_pred_loss /= len(train_loader.dataset)
                epoch_cons_loss /= len(train_loader.dataset)
                
                # Validation evaluation
                model.eval()
                val_hits_1step = 0
                val_hits_2step = 0
                val_preds_list_1step = []
                val_preds_list_2step = []
                
                with torch.no_grad():
                    for Xv, yv, candv in val_loader:
                        Xv, yv, candv = Xv.to(device), yv.to(device), candv.to(device)
                        ft, df_c, df_s, _, _ = extract_features(Xv, mean_stats, std_stats)
                        
                        pv_1step = model.predict(Xv, ft, df_c, df_s, candv, steps=1)
                        pv_2step = model.predict(Xv, ft, df_c, df_s, candv, steps=2)
                        
                        dist_1step = torch.norm(pv_1step - yv, dim=1)
                        dist_2step = torch.norm(pv_2step - yv, dim=1)
                        
                        val_hits_1step += (dist_1step <= 0.01).sum().item()
                        val_hits_2step += (dist_2step <= 0.01).sum().item()
                        
                        val_preds_list_1step.append(pv_1step.cpu().numpy())
                        val_preds_list_2step.append(pv_2step.cpu().numpy())
                        
                val_hr_1step = val_hits_1step / len(X_val)
                val_hr_2step = val_hits_2step / len(X_val)
                
                val_preds_arr_1step = np.concatenate(val_preds_list_1step, axis=0)
                val_preds_arr_2step = np.concatenate(val_preds_list_2step, axis=0)
                
                print(f"  Epoch {epoch:2d}/{args.epochs} | Loss: {epoch_loss:.4f} (CFM: {epoch_cfm_loss:.6f}, Pred: {epoch_pred_loss:.4f}, Cons: {epoch_cons_loss:.6f}) | Val 1-Step: {val_hr_1step * 100:.3f}% | Val 2-Step: {val_hr_2step * 100:.3f}%")
                
                if val_hr_2step >= best_val_hr_2step:
                    best_val_hr_2step = val_hr_2step
                    best_epoch = epoch
                    model_path = models_dir / f"model_fold_{fold}.pt"
                    torch.save(model.state_dict(), model_path)
                    best_val_preds_1step = val_preds_arr_1step
                    best_val_preds_2step = val_preds_arr_2step
                    
            print(f"Fold {fold + 1} Training Finished. Best Val Hit@1cm (2-Step): {best_val_hr_2step * 100:.3f}% at Epoch {best_epoch}")
            oof_preds_1step[val_mask] = best_val_preds_1step
            oof_preds_2step[val_mask] = best_val_preds_2step
            fold_hr_scores_1step.append(val_hr_1step)
            fold_hr_scores_2step.append(best_val_hr_2step)
            
            send_discord_notification(
                None,
                f"📢 Fold {fold + 1}/{args.folds} Finished | Best Val Hit@1cm (2-Step): {best_val_hr_2step * 100:.3f}% (Epoch {best_epoch})"
            )
            
        # Overall OOF calculation
        overall_hits_1step = np.sum(np.linalg.norm(oof_preds_1step - y_train, axis=1) <= 0.01)
        overall_hr_1step = overall_hits_1step / len(y_train)
        
        overall_hits_2step = np.sum(np.linalg.norm(oof_preds_2step - y_train, axis=1) <= 0.01)
        overall_hr_2step = overall_hits_2step / len(y_train)
        
        print(f"\n==========================================")
        print(f"Overall 5-Fold OOF Hit Rate@1cm (1-Step): {overall_hr_1step * 100:.3f}%")
        print(f"Overall 5-Fold OOF Hit Rate@1cm (2-Step): {overall_hr_2step * 100:.3f}%")
        print(f"==========================================")
        
        np.save("step57_spatiotemporal_ai/data/oof_preds_cfm_1step.npy", oof_preds_1step)
        np.save("step57_spatiotemporal_ai/data/oof_preds_cfm_2step.npy", oof_preds_2step)
        print("Saved OOF predictions for Step 57.")
        
        success_msg = (
            f"✅ Finished: [Step 57 train_automl.py] Spatiotemporal AI Model Training Complete!\n"
            f"Overall OOF Hit Rate@1cm (1-Step): **{overall_hr_1step * 100:.3f}%**\n"
            f"Overall OOF Hit Rate@1cm (2-Step): **{overall_hr_2step * 100:.3f}%**\n"
            f"Fold scores (2-Step): {[round(s*100, 3) for s in fold_hr_scores_2step]}"
        )
        send_discord_notification(None, success_msg)
        
    except Exception as e:
        error_msg = f"❌ Failed: [Step 57 train_automl.py]\nError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900])
        sys.exit(1)

if __name__ == "__main__":
    main()
