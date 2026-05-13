import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from dataset import MosquitoDatasetStep5
from model import SpectralEqMotion

def train(epochs=100, batch_size=128, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, ".."))
    train_dir = os.path.join(project_root, "data/open/train")
    labels_path = os.path.join(project_root, "data/open/train_labels.csv")
    
    if not os.path.exists(train_dir):
        train_dir = "data/open/train"
        labels_path = "data/open/train_labels.csv"
        
    save_dir = os.path.join(project_root, "outputs/step5/checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    full_dataset = MosquitoDatasetStep5(train_dir, labels_path)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = SpectralEqMotion(fft_dim=36).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                              steps_per_epoch=len(train_loader), 
                                              epochs=epochs)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            coords = batch['coords'].to(device)
            fft = batch['fft'].to(device)
            target = batch['target'].to(device)
            
            optimizer.zero_grad()
            residual_pred = model(coords, fft)
            loss = criterion(residual_pred, target)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                coords = batch['coords'].to(device)
                fft = batch['fft'].to(device)
                target = batch['target'].to(device)
                
                residual_pred = model(coords, fft)
                loss = criterion(residual_pred, target)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            print("Saved best model.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
