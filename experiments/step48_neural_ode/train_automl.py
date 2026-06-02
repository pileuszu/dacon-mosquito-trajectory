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

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.notifier import send_discord_notification
from model import SimpleNeuralODEModel, extract_features

class MosquitoSimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def stable_fold_id(sample_id: str, folds: int) -> int:
    digest = hashlib.md5(sample_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % folds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-3, help="Weight decay")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--fold-limit", type=int, default=None, help="Limit number of folds to train (for dry runs)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda, cpu, auto)")
    args = parser.parse_args()
    
    # Send start notification
    send_discord_notification(
        None,
        f"🚀 Started: [Step 48 train_automl.py] Training 5-fold Neural ODE trajectory model...\n"
        f"Config: epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, fold_limit={args.fold_limit}"
    )
    
    try:
        # Determine device
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(args.device)
        print(f"Using device: {device}")
        
        # Load dataset
        data_dir = Path("step48_neural_ode/data")
        X_train = np.load(data_dir / "train_x.npy")
        y_train = np.load(data_dir / "train_y.npy")
        
        with open(data_dir / "train_ids.json", "r") as f:
            train_ids = json.load(f)
            
        print(f"Loaded train data. X_train: {X_train.shape}, y_train: {y_train.shape}, train_ids: {len(train_ids)}")
        
        # Compute stable fold ids
        fold_ids = np.asarray([stable_fold_id(sid, args.folds) for sid in train_ids])
        
        oof_preds = np.zeros_like(y_train)
        models_dir = Path("step48_neural_ode/models")
        models_dir.mkdir(exist_ok=True)
        
        fold_hr_scores = []
        
        folds_to_train = args.folds if args.fold_limit is None else args.fold_limit
        
        for fold in range(folds_to_train):
            print(f"\n--- FOLD {fold + 1} / {args.folds} ---")
            train_mask = fold_ids != fold
            val_mask = fold_ids == fold
            
            X_tr, y_tr = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_train[val_mask], y_train[val_mask]
            
            train_ds = MosquitoSimpleDataset(X_tr, y_tr)
            val_ds = MosquitoSimpleDataset(X_val, y_val)
            
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
            
            # Compute feature scaling stats using training set
            train_coords_tensor = torch.tensor(X_tr, dtype=torch.float32).to(device)
            with torch.no_grad():
                _, _, _, _, _, _, _, _, _, mean_stats, std_stats = extract_features(train_coords_tensor)
                
            # Save scaling stats for inference later
            stats_path = models_dir / f"stats_fold_{fold}.pt"
            torch.save({"mean": mean_stats.cpu(), "std": std_stats.cpu()}, stats_path)
            print(f"Saved scaling statistics to {stats_path}")
            
            model = SimpleNeuralODEModel(input_dim=24, latent_dim=64).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            best_val_hr = 0.0
            best_epoch = 0
            
            for epoch in range(1, args.epochs + 1):
                # Ensure model.train() is called at the start of each epoch
                model.train()
                epoch_loss = 0.0
                
                for Xb, yb in train_loader:
                    Xb, yb = Xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    
                    # Extract 24-dimensional features & rotation matrix
                    ft, df, plt_, tht, _, _, _, Rt, spt, _, _ = extract_features(Xb, mean_stats, std_stats)
                    pred = model(ft, df, plt_, tht, spt, Rt)
                    
                    # 1. Huber Loss + Soft-Hit Loss (1cm boundary approximation)
                    d = torch.norm(pred - yb, dim=1)
                    soft_hit = (1 - torch.sigmoid(-(d - 0.011178) * 332.259)).mean()
                    huber = F.huber_loss(pred, yb, delta=0.001026)
                    
                    # 2. Acceleration Regularization Loss
                    reg = 0.0
                    if hasattr(model, "_last_accels") and model._last_accels:
                        reg = sum(a.pow(2).sum(-1).mean() for a in model._last_accels) / len(model._last_accels)
                        
                    loss = soft_hit + 126.309 * huber + 1e-4 * reg
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * len(Xb)
                    
                epoch_loss /= len(train_loader.dataset)
                
                # Validation evaluation
                model.eval()
                val_hits = 0
                val_preds_list = []
                
                with torch.no_grad():
                    for Xv, yv in val_loader:
                        Xv, yv = Xv.to(device), yv.to(device)
                        ft, df, plt_, tht, _, _, _, Rt, spt, _, _ = extract_features(Xv, mean_stats, std_stats)
                        pv = model(ft, df, plt_, tht, spt, Rt)
                        
                        dist = torch.norm(pv - yv, dim=1)
                        val_hits += (dist <= 0.01).sum().item()
                        val_preds_list.append(pv.cpu().numpy())
                        
                val_hr = val_hits / len(X_val)
                val_preds_arr = np.concatenate(val_preds_list, axis=0)
                
                print(f"  Epoch {epoch:2d}/{args.epochs} | Train Loss: {epoch_loss:.6f} | Val Hit@1cm: {val_hr * 100:.3f}%")
                
                # Save best checkpoint
                if val_hr >= best_val_hr:
                    best_val_hr = val_hr
                    best_epoch = epoch
                    model_path = models_dir / f"model_fold_{fold}.pt"
                    torch.save(model.state_dict(), model_path)
                    # Cache the best predictions for OOF
                    best_val_preds = val_preds_arr
                    
            print(f"Fold {fold + 1} Training Finished. Best Val Hit@1cm: {best_val_hr * 100:.3f}% at Epoch {best_epoch}")
            oof_preds[val_mask] = best_val_preds
            fold_hr_scores.append(best_val_hr)
            
            # Send fold intermediate notification if not dry run
            if args.epochs > 1:
                send_discord_notification(
                    None,
                    f"📢 Fold {fold + 1}/{args.folds} Finished | Best Val Hit@1cm: {best_val_hr * 100:.3f}% (Epoch {best_epoch})"
                )
                
        # Overall OOF calculation
        if args.fold_limit is None:
            overall_hits = np.sum(np.linalg.norm(oof_preds - y_train, axis=1) <= 0.01)
            overall_hr = overall_hits / len(y_train)
            print(f"\n==========================================")
            print(f"Overall 5-Fold OOF Hit Rate@1cm: {overall_hr * 100:.3f}%")
            print(f"==========================================")
            
            # Save OOF predictions
            np.save("step48_neural_ode/oof_predictions.npy", oof_preds)
            print("Saved OOF predictions to step48_neural_ode/oof_predictions.npy")
            
            send_discord_notification(
                None,
                f"✅ Finished: [Step 48 train_automl.py] Neural ODE Model Training Completed Successfully!\n"
                f"Overall OOF Hit Rate@1cm: **{overall_hr * 100:.3f}%**\n"
                f"Fold scores: {[round(s*100, 3) for s in fold_hr_scores]}"
            )
        else:
            print(f"\nDry run completed for {args.fold_limit} fold(s).")
            send_discord_notification(
                None,
                f"✅ Finished: [Step 48 train_automl.py] Dry Run Completed Successfully!\n"
                f"Fold scores: {[round(s*100, 3) for s in fold_hr_scores]}"
            )
            
    except Exception as e:
        error_msg = f"❌ Failed: [Step 48 train_automl.py]\nError: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        send_discord_notification(None, error_msg[:1900]) # Discord 2000 char limit
        sys.exit(1)

if __name__ == "__main__":
    main()
